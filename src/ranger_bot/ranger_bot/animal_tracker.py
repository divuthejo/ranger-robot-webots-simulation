"""Where are the animals? Camera boxes + lidar -> animal positions on the map, tracked over time.

    /ranger/detections      (in)  vision_msgs/Detection2DArray from animal_detector.py
    /scan                   (in)  LDS-01 (Webots order: clockwise, negative angle_increment)
    /ranger/animals         (out) std_msgs/String JSON, latched, 1 Hz: every known track
    /ranger/animal_markers  (out) visualization_msgs/MarkerArray for RViz
    /ranger/animal_cloud    (out) sensor_msgs/PointCloud2 (map): each freshly seen animal's
                                  whole body as a filled disc -> Nav2 costmaps, because the
                                  lidar only sees a horse's legs and Nav2 drove between them
    sightings/locations.csv       one line when a track is confirmed or re-found

Range to an animal, from the box:
  * lidar: the nearest return whose direction (seen from the camera) falls inside the middle
    of the box - accurate, but the LDS-01 plane is 0.173 m up, so rabbits/cats/foxes often
    sit under it, and at range a box can hold background trees;
  * ground plane: the camera is level and 0.21 m up, so the row of the box bottom (the feet)
    gives the distance: d = h * f / (v - cy). Good close, coarse far (1 px ~ 0.25 m at 5 m).
  The lidar range wins when the two agree (within 35 %), else the ground estimate is used.
The point is then pushed back by part of the animal's radius (lidar sees its near side).

Tracks: per class, gated association (animals walk), exponential smoothing, confirmed after
3 hits; a confirmed track is kept with its last-seen time even when out of view (that is
what tasks.py uses for "go towards the deer"). 'person' tracks feed intruder.py.
"""

import csv
import json
import math
import os
import time

CAM_FOV = 1.2
CAM_W, CAM_H_PX = 640, 480
CAM_HEIGHT = 0.363              # m above the ground: the .wbt Pose z 0.21 is relative to the
                                # robot origin, which sits 0.153 m up (gps z) - using 0.21 made
                                # every range ~half the truth (measured 2026-11-01)
CAM_X = 0.03                    # m ahead of base_link
LIDAR_X = -0.03                 # m (base_link_LDS-01_joint)
FX = (CAM_W / 2) / math.tan(CAM_FOV / 2)
CY = CAM_H_PX / 2
MAX_RANGE = 9.0
MAX_GROUND_RANGE = 7.0          # beyond this 1 px of box bottom is > 0.3 m of range
MAX_TURN_RATE = 0.35            # rad/s: skip frames while turning (boxes smear, pose lags)
RADIUS = {'deer': 0.6, 'horse': 0.9, 'cow': 0.9, 'sheep': 0.5, 'dog': 0.4, 'fox': 0.35,
          'rabbit': 0.2, 'cat': 0.25, 'person': 0.25}


def bearing(u):
    """Image column -> bearing from the camera axis (rad, left positive)."""
    return math.atan2(CAM_W / 2 - u, FX)


def ground_range(v_bottom):
    """Box bottom row -> distance (m) from the camera along the ground, or None."""
    dv = v_bottom - CY
    if dv < 2.0 or v_bottom >= CAM_H_PX - 2:     # at/above the horizon, or cut off by the edge
        return None
    return CAM_HEIGHT * FX / dv


def lidar_range(box, beams, shrink=0.25):
    """Nearest lidar return inside the middle of the box, as distance from the camera.

    beams: [(angle in base_link, range)] valid returns. box: (u1, v1, u2, v2).
    """
    u1, _, u2, _ = box
    w = u2 - u1
    left, right = bearing(u1 + shrink * w), bearing(u2 - shrink * w)
    best = None
    for a, r in beams:
        px, py = LIDAR_X + r * math.cos(a) - CAM_X, r * math.sin(a)     # from the camera
        b = math.atan2(py, px)
        if right <= b <= left:
            d = math.hypot(px, py)
            if best is None or d < best:
                best = d
    return best


PERSON_HEIGHT = 1.7             # m: a person's known height gives range from box height
MAX_PERSON_RANGE = 14.0


def locate(cls, box, beams, pose):
    """Map (x, y), the range used and its source - or None when no range is usable.

    Lidar is accepted only when it agrees with the ground-plane range, or when the box is cut
    off at the image bottom (very close, no ground estimate). A narrow box (a person behind a
    tree) let a lidar return off the trunk in front through - 2.5 m for a person 11 m away.
    People: their known height gives a second range (1.7 m * f / box px), averaged with the
    ground estimate; that also works far beyond the 7 m ground-plane limit.
    """
    u1, v1, u2, v2 = box
    g = ground_range(v2)
    cut_off = v2 >= CAM_H_PX - 2
    lid = lidar_range(box, beams)
    if lid is not None and ((g is not None and abs(lid - g) <= 0.35 * g) or
                            (g is None and cut_off)):
        rng, src = lid + 0.6 * RADIUS.get(cls, 0.3), 'lidar'     # near side -> centre
    elif cls == 'person' and not cut_off and v2 - v1 > 10:
        hr = PERSON_HEIGHT * FX / (v2 - v1)
        rng = (hr + g) / 2 if g is not None and abs(hr - g) <= 0.5 * hr else hr
        if rng > MAX_PERSON_RANGE:
            return None
        src = 'height'
    elif g is not None and g <= MAX_GROUND_RANGE:
        rng, src = g + 0.3 * RADIUS.get(cls, 0.3), 'ground'      # feet -> centre (roughly)
    else:
        return None
    if rng > (MAX_PERSON_RANGE if cls == 'person' else MAX_RANGE):
        return None
    b = bearing((u1 + u2) / 2)
    x, y, yaw = pose
    cx, cy = x + CAM_X * math.cos(yaw), y + CAM_X * math.sin(yaw)
    return cx + rng * math.cos(yaw + b), cy + rng * math.sin(yaw + b), rng, src


def body_points(tracks, max_age=8.0, step=0.07, robot=None, keep_clear=0.35):
    """(x, y, z) points filling a disc of each fresh track's body radius - none within
    keep_clear of the robot: a slightly-off track over the robot would make Nav2 think it is
    inside an obstacle and refuse to move at all."""
    pts = []
    for t in tracks.values():
        if t['age_s'] > max_age:
            continue
        r = RADIUS.get(t['cls'], 0.3)
        n = int(r / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if (i * step) ** 2 + (j * step) ** 2 <= r * r:
                    x, y = t['x'] + i * step, t['y'] + j * step
                    if robot and math.hypot(x - robot[0], y - robot[1]) < keep_clear:
                        continue
                    pts.append((x, y, 0.2))
    return pts


class Tracker:
    """Per-class tracks of located detections. ROS-free (unit-tested)."""

    def __init__(self, gate=2.0, alpha=0.4, confirm=3, confirm_window=4.0, drop_tentative=6.0):
        self.gate, self.alpha, self.confirm = gate, alpha, confirm
        self.confirm_window, self.drop_tentative = confirm_window, drop_tentative
        self.tracks = []                    # dicts
        self.next_id = 1

    def update(self, cls, x, y, score, now, src='lidar'):
        """Add one located detection. Returns ('confirmed'|'refound'|None, track)."""
        best, bd = None, self.gate
        for t in self.tracks:
            if t['cls'] == cls:
                d = math.hypot(t['x'] - x, t['y'] - y)
                if d < bd:
                    best, bd = t, d
        if best is None:
            t = {'id': self.next_id, 'cls': cls, 'x': x, 'y': y, 'hits': 1, 'first': now,
                 'last': now, 'score': score, 'confirmed': False, 'src': src}
            self.next_id += 1
            self.tracks.append(t)
            return None, t
        t = best
        gap = now - t['last']
        a = self.alpha if src == 'lidar' else self.alpha * 0.6   # trust lidar fixes more
        t['x'] += a * (x - t['x'])
        t['y'] += a * (y - t['y'])
        t['hits'] += 1
        t['last'], t['score'], t['src'] = now, max(t['score'] * 0.9, score), src
        if not t['confirmed'] and t['hits'] >= self.confirm and \
                now - t['first'] <= self.confirm_window + gap:
            t['confirmed'] = True
            return 'confirmed', t
        if t['confirmed'] and gap > 60.0:
            return 'refound', t
        return None, t

    def prune(self, now):
        self.tracks = [t for t in self.tracks
                       if t['confirmed'] or now - t['last'] <= self.drop_tentative]
        # two confirmed tracks of one class that drifted together are one animal
        keep = []
        for t in sorted(self.tracks, key=lambda t: -t['hits']):
            if t['confirmed'] and any(k['cls'] == t['cls'] and k['confirmed'] and
                                      math.hypot(k['x'] - t['x'], k['y'] - t['y']) < 1.0
                                      for k in keep):
                continue
            keep.append(t)
        self.tracks = keep

    def snapshot(self, now):
        """{name: {...}} of confirmed tracks; 'deer', 'deer 2', ... by age."""
        out, count = {}, {}
        for t in sorted((t for t in self.tracks if t['confirmed']), key=lambda t: t['first']):
            count[t['cls']] = count.get(t['cls'], 0) + 1
            name = t['cls'] if count[t['cls']] == 1 else f"{t['cls']} {count[t['cls']]}"
            out[name] = {'cls': t['cls'], 'x': round(t['x'], 2), 'y': round(t['y'], 2),
                         'age_s': round(now - t['last'], 1), 'hits': t['hits'],
                         'confidence': round(t['score'], 2), 'source': t['src'],
                         'id': t['id']}
        return out


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan, PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener
    from vision_msgs.msg import Detection2DArray
    from visualization_msgs.msg import Marker, MarkerArray

    class AnimalTracker(Node):
        def __init__(self):
            super().__init__('animal_tracker')
            self.min_score = self.declare_parameter('min_score', 0.5).value
            # people: the zero-shot model scores the intruder only ~0.2 close up (head cut off);
            # the detector's shape/height check is what filters false people
            self.person_min_score = self.declare_parameter('person_min_score', 0.2).value
            self.dir = os.path.expanduser(self.declare_parameter(
                'sightings_dir', '~/ranger_ws/sightings').value)
            os.makedirs(self.dir, exist_ok=True)
            self.tracker = Tracker()
            self.tf = Buffer()
            TransformListener(self.tf, self)
            self.scan = None
            self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m),
                                     qos_profile_sensor_data)
            self.create_subscription(Detection2DArray, 'ranger/detections', self.on_dets, 10)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub = self.create_publisher(String, 'ranger/animals', latched)
            self.mark_pub = self.create_publisher(MarkerArray, 'ranger/animal_markers', 1)
            self.cloud_pub = self.create_publisher(PointCloud2, 'ranger/animal_cloud', 1)
            self.create_timer(1.0, self.publish)
            self.create_timer(0.25, self.publish_cloud)
            self.get_logger().info('Animal tracker ready -> /ranger/animals')

        def pose(self):
            try:
                t = self.tf.lookup_transform('map', 'base_link', Time())
            except TransformException:
                return None
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw

        def beams(self):
            s = self.scan
            if s is None:
                return []
            out = []
            for i, r in enumerate(s.ranges):
                if s.range_min < r < s.range_max and math.isfinite(r):
                    out.append((s.angle_min + i * s.angle_increment, r))
            return out

        def pose_at(self, stamp):
            """Robot map pose when the image was taken (it was ~0.3 s ago: using the latest
            pose put bearings 7 deg off while turning), and the turn rate around then."""
            from rclpy.duration import Duration
            t = Time.from_msg(stamp)
            if t.nanoseconds < 400_000_000:             # first moments of sim time
                return None, None
            try:
                a = self.tf.lookup_transform('map', 'base_link', t, timeout=Duration(seconds=0.2))
                b = self.tf.lookup_transform('map', 'base_link', t - Duration(seconds=0.3))
            except (TransformException, ValueError):
                return None, None
            yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            ya, yb = yaw(a.transform.rotation), yaw(b.transform.rotation)
            rate = abs(math.atan2(math.sin(ya - yb), math.cos(ya - yb))) / 0.3
            return (a.transform.translation.x, a.transform.translation.y, ya), rate

        def on_dets(self, msg):
            pose, rate = self.pose_at(msg.header.stamp)
            if pose is None or rate > MAX_TURN_RATE:
                return
            beams = None
            now = time.time()
            for d in msg.detections:
                if not d.results:
                    continue
                h = d.results[0].hypothesis
                if h.score < (self.person_min_score if h.class_id == 'person' else self.min_score):
                    continue
                cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
                box = (cx - d.bbox.size_x / 2, cy - d.bbox.size_y / 2,
                       cx + d.bbox.size_x / 2, cy + d.bbox.size_y / 2)
                if beams is None:
                    beams = self.beams()
                loc = locate(h.class_id, box, beams, pose)
                if loc is None:
                    continue
                x, y, rng, src = loc
                event, t = self.tracker.update(h.class_id, x, y, h.score, now, src)
                if event:
                    self.log_location(event, t, rng, pose)
            self.tracker.prune(now)

        def log_location(self, event, t, rng, pose):
            path = os.path.join(self.dir, 'locations.csv')
            row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': event, 'cls': t['cls'],
                   'track': t['id'], 'x': round(t['x'], 2), 'y': round(t['y'], 2),
                   'range_m': round(rng, 2), 'source': t['src'],
                   'robot': json.dumps([round(v, 2) for v in pose])}
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            self.get_logger().info(f"{t['cls']} {event} at ({t['x']:.1f}, {t['y']:.1f}) "
                                   f"[{t['src']}, {rng:.1f} m away]")
            self.publish()

        def publish_cloud(self):
            from std_msgs.msg import Header
            h = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
            pts = body_points(self.tracker.snapshot(time.time()), robot=self.pose())
            self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(h, pts))

        def publish(self):
            now = time.time()
            snap = self.tracker.snapshot(now)
            self.pub.publish(String(data=json.dumps(snap)))
            ma = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            ma.markers.append(clear)
            for k, (name, t) in enumerate(snap.items()):
                fresh = t['age_s'] < 10
                person = t['cls'] == 'person'
                for j, kind in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
                    m = Marker()
                    m.header.frame_id = 'map'
                    m.ns, m.id, m.type = 'animals', 2 * k + j, kind
                    m.pose.position.x, m.pose.position.y = t['x'], t['y']
                    m.pose.orientation.w = 1.0
                    if kind == Marker.CYLINDER:
                        r = RADIUS.get(t['cls'], 0.3)
                        m.scale.x = m.scale.y = 2 * r
                        m.scale.z = 0.1
                        m.pose.position.z = 0.05
                        m.color.r, m.color.g, m.color.b = (1.0, 0.1, 0.1) if person else \
                            (0.1, 0.8, 0.2)
                        m.color.a = 0.9 if fresh else 0.35
                    else:
                        m.pose.position.z = 0.8
                        m.scale.z = 0.35
                        m.text = f"{name} ({t['age_s']:.0f}s)" if not fresh else name
                        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                    ma.markers.append(m)
            self.mark_pub.publish(ma)

    rclpy.init(args=args)
    node = AnimalTracker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
