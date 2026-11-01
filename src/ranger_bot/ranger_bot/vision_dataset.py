"""Collect and auto-label camera frames of the reserve's animals, for training the detector.

    ros2 run ranger_bot vision_dataset collect   # drive rings around every animal (~30 min)
    ros2 run ranger_bot vision_dataset label     # YOLO-World boxes + names from geometry
    ros2 run ranger_bot vision_dataset export-person   # zero-shot 'person' model (intruders)

collect: with navigation.launch.py running, the robot visits standing spots on two rings
(near/far) around each animal, facing it and turned a little left and right, and saves
each camera frame with the robot's map pose (frames.jsonl next to the images).

label: YOLO-World finds the boxes (it reliably finds *an* animal, but mixes up the sim
models - deer/horse/cow, fox/cat/dog). The name comes from geometry instead: the box's
bearing from the robot's pose is matched against the animal positions in the world file.
Writes an Ultralytics dataset (images/, labels/, data.yaml) split train/val by ring spot,
so near-duplicate frames never straddle the split.
"""

import json
import math
import os
import random
import shutil
import sys
import time

DATA = os.path.expanduser('~/ranger_ws/datasets/animals')
WORLD = os.path.expanduser('~/ranger_ws/src/ranger_bot/worlds/nature_reserve.wbt')
CLASSES = ['deer', 'horse', 'cow', 'sheep', 'fox', 'rabbit', 'dog', 'cat']

CAM_FOV = 1.2               # rad, horizontal (Camera.fieldOfView in the .wbt)
CAM_W, CAM_H = 640, 480
CAM_X = 0.03                # m ahead of base_link
RINGS = (1.0, 2.2)          # m from the animal's edge
ANGLES = 8                  # spots per ring
YAW_OFFSETS = (0.0, -0.35, 0.35)   # rad: animal centred, then right / left of centre


def bearing_of_pixel(u):
    """Image column -> bearing (rad, left positive) relative to the camera axis."""
    fx = (CAM_W / 2) / math.tan(CAM_FOV / 2)
    return math.atan2(CAM_W / 2 - u, fx)


def visible_animals(pose, animals, max_range=12.0):
    """[(distance, bearing, name)] of known animals inside the camera's view, nearest first."""
    x, y, yaw = pose
    cx, cy = x + CAM_X * math.cos(yaw), y + CAM_X * math.sin(yaw)
    seen = []
    for name, (ax, ay, _) in animals.items():
        rel = math.atan2(ay - cy, ax - cx) - yaw
        rel = math.atan2(math.sin(rel), math.cos(rel))
        dist = math.hypot(ax - cx, ay - cy)
        if abs(rel) <= CAM_FOV / 2 + 0.1 and dist <= max_range:
            seen.append((dist, rel, name))
    return sorted(seen)


def match_animal(u1, u2, pose, animals, margin=math.radians(4)):
    """Name of the nearest known animal whose bearing lies within box columns u1..u2.

    Bearing *inside the box* (not near its centre): close up an animal spans ~30 deg and the
    localisation error (~0.15 m) moves its bearing by several degrees.
    """
    left, right = bearing_of_pixel(min(u1, u2)) + margin, bearing_of_pixel(max(u1, u2)) - margin
    for _, rel, name in visible_animals(pose, animals):
        if right <= rel <= left:
            return name
    return None


def inside(a, b, frac=0.8):
    """Is box a (x1, y1, x2, y2) mostly (frac of its area) inside box b?"""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return area > 0 and w * h / area >= frac


def ring_spots(animal, rings=RINGS, angles=ANGLES):
    """[(ring, k, x, y, yaw_facing)] around animal=(x, y, radius), walking round each ring."""
    ax, ay, r = animal
    spots = []
    for ring in rings:
        for k in range(angles):
            a = 2 * math.pi * k / angles
            x, y = ax + (r + ring) * math.cos(a), ay + (r + ring) * math.sin(a)
            spots.append((ring, k, x, y, math.atan2(ay - y, ax - x)))
    return spots


def load_animals():
    from ranger_bot.tasks import parse_animals
    with open(WORLD) as f:
        return parse_animals(f.read())


# ---------------------------------------------------------------- collect

def collect():
    import cv2
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    from rclpy.parameter import Parameter
    nav = BasicNavigator(node_name='vision_dataset')
    nav.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    frame, costmap = [None], [None]
    nav.create_subscription(Image, '/camera/image_color', lambda m: frame.__setitem__(0, m), 1)
    nav.create_subscription(OccupancyGrid, 'global_costmap/costmap',
                            lambda m: costmap.__setitem__(0, m),
                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                       reliability=ReliabilityPolicy.RELIABLE))
    tf = Buffer()
    TransformListener(tf, nav)
    mode = nav.create_publisher(String, 'ranger/set_mode', 10)
    patrol = nav.create_publisher(String, 'ranger/patrol', 10)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(nav, timeout_sec=0.05)

    def free(x, y):
        m = costmap[0]
        if m is None:
            return True
        i = int((x - m.info.origin.position.x) / m.info.resolution)
        j = int((y - m.info.origin.position.y) / m.info.resolution)
        if not (0 <= i < m.info.width and 0 <= j < m.info.height):
            return False
        return 0 <= m.data[j * m.info.width + i] <= 50

    def pose():
        try:
            t = tf.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def go(x, y, yaw, timeout=None):
        if timeout is None:                     # ~0.1 m/s worst case + turning, not a fixed 60 s
            here = pose()
            dist = math.hypot(x - here[0], y - here[1]) if here else 20.0
            timeout = 30.0 + dist / 0.1
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.header.stamp = nav.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        nav.goToPose(p)
        start = last_check = last_move = time.time()
        reseeds = 0
        anchor = truth()
        while not nav.isTaskComplete():
            spin(0.1)
            if time.time() - start > timeout:
                nav.cancelTask()
                return False
            t = truth()
            if t and anchor and math.hypot(t[0] - anchor[0], t[1] - anchor[1]) > 0.1:
                anchor, last_move = t, time.time()
            elif t and time.time() - last_move > 60.0:      # turn-on-the-spot goals take < 20 s
                print(f'  stuck at ({t[0]:.2f}, {t[1]:.2f}) for 60 s - giving up this goal', flush=True)
                nav.cancelTask()
                return False
            # Check AMCL against ground truth WHILE driving too: a 1.6 m drift mid-drive steered
            # the robot blind onto lake rock 1 (lidar can't see it), wedged in its keepout zone.
            if time.time() - last_check > 2.0:
                last_check = time.time()
                if not check_localisation():
                    nav.cancelTask()
                    reseeds += 1
                    if reseeds > 3:
                        return False
                    p.header.stamp = nav.get_clock().now().to_msg()
                    nav.goToPose(p)
        return nav.getResult() == TaskResult.SUCCEEDED

    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PointStamped
    gps = [None]
    nav.create_subscription(PointStamped, '/gps', lambda m: gps.__setitem__(0, m.point),
                            qos_profile_sensor_data)

    from sensor_msgs.msg import Imu
    imu_yaw = [None]

    def on_imu(m):
        q = m.orientation
        imu_yaw[0] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    nav.create_subscription(Imu, '/imu', on_imu, qos_profile_sensor_data)

    def truth():
        """Simulator ground truth (gps x,y + InertialUnit yaw), base_link pose in the world."""
        if gps[0] is None or imu_yaw[0] is None:
            return None
        return gps[0].x, gps[0].y, imu_yaw[0]

    from geometry_msgs.msg import PoseWithCovarianceStamped
    initialpose = nav.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

    def check_localisation(max_xy=0.3, max_yaw=0.1):
        """AMCL vs ground truth. AMCL drifts in open grass (the dog at 10,2 is in it), which put
        the camera on the wrong patch of grass and labelled boxes with the wrong animal.
        If it is off, re-seed AMCL at the true pose (sim time stamp!) and return False."""
        p, t = pose(), truth()
        if p is None or t is None:
            return True
        dxy = math.hypot(p[0] - t[0], p[1] - t[1])
        dyaw = math.remainder(p[2] - t[2], 2 * math.pi)
        if dxy <= max_xy and abs(dyaw) <= max_yaw:
            return True
        print(f'  localisation off by {dxy:.2f} m / {math.degrees(dyaw):+.0f} deg '
              f'-> re-seeding AMCL from ground truth', flush=True)
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.header.stamp = nav.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = t[0], t[1]
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(t[2] / 2), math.cos(t[2] / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.05 ** 2
        m.pose.covariance[35] = 0.03 ** 2
        initialpose.publish(m)
        spin(2.0)
        return False

    def in_reserve():
        """Ground truth sanity check: a physics blow-up flings the robot far away."""
        p = gps[0]
        return p is None or (abs(p.x) < 20 and abs(p.y) < 20 and abs(p.z) < 1)

    raw = os.path.join(DATA, 'raw')
    os.makedirs(raw, exist_ok=True)
    log = open(os.path.join(raw, 'frames.jsonl'), 'a')
    # Wait for discovery, or the first messages are silently lost - which left the patrol
    # running and fighting these goals in the first run.
    start = time.time()
    while (patrol.get_subscription_count() == 0 or mode.get_subscription_count() == 0) \
            and time.time() - start < 10:
        spin(0.2)
    patrol.publish(String(data='pause'))
    spin(1.0)
    mode.publish(String(data='auto'))
    spin(2.0)
    animals = load_animals()
    only = sys.argv[2:] or list(animals)
    saved = 0
    for name in only:
        rings = tuple(float(r) for r in os.environ.get('RINGS', '').split(',') if r) or RINGS
        angles = int(os.environ.get('ANGLES', ANGLES))
        spots = [s for s in ring_spots(animals[name], rings, angles) if free(s[2], s[3])]
        print(f'{name}: {len(spots)} free spots', flush=True)
        for ring, k, x, y, face in spots:
            if not in_reserve():
                print(f'ABORT: robot left the reserve (gps {gps[0].x:.0f}, {gps[0].y:.0f}, '
                      f'{gps[0].z:.0f}) - physics blow-up, restart the sim', flush=True)
                mode.publish(String(data='stop'))
                spin(0.5)
                rclpy.shutdown()
                return
            check_localisation()
            if not go(x, y, face):
                print(f'  ring {ring} spot {k}: unreachable', flush=True)
                continue
            for off in YAW_OFFSETS:
                if off and not go(x, y, face + off, timeout=20.0):
                    continue
                spin(1.0)                               # settle; fresh frame
                if not check_localisation():            # was off: we are not where we think
                    go(x, y, face + off)
                    spin(1.0)
                m, p, t = frame[0], pose(), truth()
                if m is None or p is None:
                    continue
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
                fn = f'{name}_r{ring}_a{angles}k{k}_o{off:+.2f}.jpg'
                cv2.imwrite(os.path.join(raw, fn), img)
                rec = {'file': fn, 'target': name, 'spot': f'{name}_{ring}_{angles}_{k}',
                       'pose': [round(v, 4) for v in p]}
                if t is not None:
                    rec['truth'] = [round(v, 4) for v in t]
                log.write(json.dumps(rec) + '\n')
                log.flush()
                saved += 1
        print(f'  {name} done, {saved} frames so far', flush=True)
    mode.publish(String(data='stop'))
    spin(0.5)
    print(f'Collected {saved} frames in {raw}')
    rclpy.shutdown()


# ---------------------------------------------------------------- label

def label(val_fraction=0.15, conf=0.1):
    from ultralytics import YOLOWorld

    raw = os.path.join(DATA, 'raw')
    frames = [json.loads(line) for line in open(os.path.join(raw, 'frames.jsonl'))]
    if os.environ.get('TRUTH_ONLY', '1') == '1':
        # Frames from before 2026-11-01 have only the AMCL pose, which drifted enough to put
        # the wrong animal's name on boxes. Ground-truth frames only, unless TRUTH_ONLY=0.
        n = len(frames)
        frames = [f for f in frames if 'truth' in f]
        print(f'{len(frames)} of {n} frames have ground truth (TRUTH_ONLY=0 to use all)')
    # A spot visited twice (re-run) wrote the same file name: keep the last record only.
    frames = list({f['file']: f for f in frames}.values())
    animals = load_animals()
    det = YOLOWorld(os.path.expanduser('~/ranger_ws/models/yolov8s-worldv2.pt'))
    det.set_classes(CLASSES + ['animal'])
    spots = sorted({f['spot'] for f in frames})
    random.Random(0).shuffle(spots)
    val_spots = set(spots[:max(1, int(len(spots) * val_fraction))])
    for split in ('train', 'val'):
        for sub in ('images', 'labels'):
            d = os.path.join(DATA, sub, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
    counts = {c: 0 for c in CLASSES}
    dropped, skipped = 0, 0
    for f in frames:
        r = det.predict(os.path.join(raw, f['file']), conf=conf, agnostic_nms=True,
                        verbose=False)[0]
        boxes = [b.xyxy[0].tolist() for b in r.boxes]
        # small, or nested in a bigger box (a far animal seen between the legs, a body part)
        boxes = [b for b in boxes if b[2] - b[0] >= 20 and b[3] - b[1] >= 20
                 and not any(o is not b and inside(b, o) for o in boxes)]
        pose = f.get('truth', f['pose'])          # ground truth when recorded
        named = [(b, match_animal(b[0], b[2], pose, animals)) for b in boxes]
        dropped += sum(n is None for _, n in named)
        named = [(b, n) for b, n in named if n]
        near = {n for d, _, n in visible_animals(pose, animals) if d <= 5.0}
        if near - {n for _, n in named}:
            skipped += 1                    # a close animal in view but not boxed: unsure frame
            continue
        lines = []
        for (x1, y1, x2, y2), name in named:
            counts[name] += 1
            lines.append(f'{CLASSES.index(name)} {(x1 + x2) / 2 / CAM_W:.5f} '
                         f'{(y1 + y2) / 2 / CAM_H:.5f} {(x2 - x1) / CAM_W:.5f} '
                         f'{(y2 - y1) / CAM_H:.5f}')
        split = 'val' if f['spot'] in val_spots else 'train'
        shutil.copy(os.path.join(raw, f['file']), os.path.join(DATA, 'images', split, f['file']))
        with open(os.path.join(DATA, 'labels', split, f['file'][:-4] + '.txt'), 'w') as out:
            out.write('\n'.join(lines) + ('\n' if lines else ''))
    with open(os.path.join(DATA, 'data.yaml'), 'w') as y:
        y.write(f'path: {DATA}\ntrain: images/train\nval: images/val\n'
                f'names: {json.dumps(dict(enumerate(CLASSES)))}\n')
    print(f'{len(frames) - skipped} of {len(frames)} frames kept ({skipped} unsure), boxes per class '
          f'{counts}, {dropped} unmatched boxes dropped')
    print(f'val spots: {len(val_spots)} of {len(spots)}  ->  {os.path.join(DATA, "data.yaml")}')


def export_person():
    """Zero-shot YOLO-World with the single class 'person' -> OpenVINO, for intruder alerts."""
    from ultralytics import YOLOWorld
    src = os.path.expanduser('~/ranger_ws/models/yolov8s-worldv2.pt')
    dst = os.path.expanduser('~/ranger_ws/models/person_openvino_model')
    m = YOLOWorld(src)
    m.set_classes(['person'])
    tmp = os.path.expanduser('~/ranger_ws/models/yolov8s-world-person.pt')
    m.save(tmp)
    out = YOLOWorld(tmp).export(format='openvino', imgsz=640)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(out, dst)
    print(f'person model -> {dst}')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'export-person':
        export_person()
        return
    if cmd == 'collect':
        collect()
    elif cmd == 'label':
        label()
    else:
        print(__doc__)
