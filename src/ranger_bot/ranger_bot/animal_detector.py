"""Animal detector: the robot's camera -> which animals are in view, and a sightings log.

    /camera/image_color   (in)  640x480 bgra8 from the Webots camera, 10 Hz
    /ranger/detections    (out) vision_msgs/Detection2DArray, every processed frame
    /camera/detections    (out) the frame with boxes drawn, for RViz (Image display)
    /ranger/sightings     (out) std_msgs/String JSON, once per confirmed sighting
    /ranger/say           (out) "Deer spotted" -> announcer.py

People: an optional second, zero-shot model (YOLO-World with the single class 'person',
exported with vision_dataset.py export-person) runs every `person_every` frames; its boxes are
added as class 'person' (intruder.py raises the alarm, so no "spotted" line for them).

Model: YOLO fine-tuned on frames of this reserve's animals (vision_dataset.py collects and
auto-labels them), exported to OpenVINO, so it runs on the laptop CPU without torch at
inference. Off-the-shelf YOLO-World only named 4/8 of the sim animals correctly.

A sighting is confirmed when the same animal is seen in `confirm` of the last `window`
processed frames (one-frame flickers are ignored); after that it is not reported again for
`cooldown` seconds. Each sighting goes to `sightings_dir`/sightings.csv with a snapshot,
the robot's map pose and the animal's bearing - the patrol report is built from this.
"""

import collections
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

CAM_FOV = 1.2               # rad, horizontal (Camera.fieldOfView in the .wbt)


class Confirmer:
    """Turn per-frame detections into sightings: seen in `confirm` of the last `window`
    frames, then quiet for `cooldown` s. ROS-free so it can be unit-tested."""

    def __init__(self, window=5, confirm=3, cooldown=60.0):
        self.window, self.confirm, self.cooldown = window, confirm, cooldown
        self.history = collections.deque(maxlen=window)
        self.last_reported = {}

    def update(self, names, now):
        """names seen in this frame -> list of names to report as new sightings."""
        self.history.append(set(names))
        new = []
        for name in set(names):
            seen = sum(name in frame for frame in self.history)
            if seen >= self.confirm and now - self.last_reported.get(name, -1e9) >= self.cooldown:
                self.last_reported[name] = now
                new.append(name)
        return new


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


CAM_HEIGHT = 0.363          # m above ground (robot origin 0.153 + camera Pose 0.21)
CAM_CY = 240.0              # horizon row: the camera is level


def implied_height(box, cam_height=CAM_HEIGHT, cy=CAM_CY):
    """Real height (m) of an object standing on the ground, from its box: the box bottom gives
    its distance (d = H f / (v - cy)), the box height then gives metres (h = px d / f)."""
    x1, y1, x2, y2 = box
    if y2 - cy < 2.0:
        return None                     # feet at/above the horizon: not standing on the ground
    return (y2 - y1) * cam_height / (y2 - cy)


def plausible_person(box, min_h=1.2, max_h=2.4, max_aspect=0.45):
    """Zero-shot 'person' boxes also land on animals (a cat at 8 m scored 0.53, a distant deer
    0.60, a horse 0.73). A standing person is 1.2-2.4 m tall - the box's implied height - AND
    narrow: width <= 0.45 x height (intruder boxes measured 0.24-0.33; animals 0.6-1.2, a cow
    from behind 0.64). Height alone passes horses and cows (~1.4 m)."""
    x1, y1, x2, y2 = box
    if (x2 - x1) > max_aspect * (y2 - y1):
        return False
    h = implied_height(box)
    return h is not None and min_h <= h <= max_h


def resolve_people(animal_dets, person_dets, max_iou=0.3):
    """Keep plausible person boxes; drop animal boxes that sit on one. The animal model never
    saw a person in training and calls the intruder a 'cow' (0.7-0.84) - with "animal wins"
    the real intruder was never reported. dets: [(box, name, score)]."""
    people = [p for p in person_dets if plausible_person(p[0])]
    animals = [a for a in animal_dets if all(iou(a[0], p[0]) <= max_iou for p in people)]
    return animals, people


def bearing(u, width):
    """Image column -> bearing (rad, left positive) relative to the camera axis."""
    fx = (width / 2) / math.tan(CAM_FOV / 2)
    return math.atan2(width / 2 - u, fx)


class AnimalDetector(Node):

    def __init__(self):
        super().__init__('animal_detector')
        model = os.path.expanduser(self.declare_parameter(
            'model', '~/ranger_ws/models/animals_openvino_model').value)
        self.conf = self.declare_parameter('confidence', 0.5).value
        rate = self.declare_parameter('rate_hz', 3.0).value
        self.sightings_dir = os.path.expanduser(self.declare_parameter(
            'sightings_dir', '~/ranger_ws/sightings').value)
        self.confirmer = Confirmer(
            window=self.declare_parameter('window', 5).value,
            confirm=self.declare_parameter('confirm', 3).value,
            cooldown=self.declare_parameter('cooldown', 60.0).value)
        from ultralytics import YOLO            # imported here so the module loads without it
        self.model = YOLO(model, task='detect')
        self.names = self.model.names
        person = os.path.expanduser(self.declare_parameter(
            'person_model', '~/ranger_ws/models/person_openvino_model').value)
        self.person_every = int(self.declare_parameter('person_every', 2).value)
        self.person_conf = self.declare_parameter('person_confidence', 0.2).value
        self.person_model = YOLO(person, task='detect') if os.path.isdir(person) else None
        self.frames = 0
        os.makedirs(self.sightings_dir, exist_ok=True)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.frame = None
        self.create_subscription(Image, '/camera/image_color',
                                 lambda m: setattr(self, 'frame', m), 1)
        self.det_pub = self.create_publisher(Detection2DArray, 'ranger/detections', 10)
        self.img_pub = self.create_publisher(Image, 'camera/detections', 1)
        self.sight_pub = self.create_publisher(String, 'ranger/sightings', 10)
        self.say_pub = self.create_publisher(String, 'ranger/say', 10)
        self.create_timer(1.0 / rate, self.process)
        self.get_logger().info(f'Animal detector ready ({len(self.names)} classes, '
                               f'{rate:g} Hz, person model '
                               f'{"on" if self.person_model else "OFF (not exported)"}, '
                               f'sightings -> {self.sightings_dir})')

    def robot_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def process(self):
        msg, self.frame = self.frame, None
        if msg is None:
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        img = np.ascontiguousarray(img)
        result = self.model.predict(img, conf=self.conf, verbose=False)[0]
        boxes = [(b.xyxy[0].tolist(), self.names[int(b.cls)], float(b.conf)) for b in result.boxes]
        self.frames += 1
        people = []
        if self.person_model is not None and self.frames % self.person_every == 0:
            pr = self.person_model.predict(img, conf=self.person_conf, verbose=False)[0]
            people = [(b.xyxy[0].tolist(), 'person', float(b.conf)) for b in pr.boxes]
            boxes, people = resolve_people(boxes, people)
            self.last_people = people
        elif getattr(self, 'last_people', None):
            # frames without the person model (half the CPU): last frame's people still
            # suppress the animal model's 'cow' box on the intruder, but are not re-reported
            boxes, _ = resolve_people(boxes, self.last_people)
        out = Detection2DArray(header=msg.header)
        found = {}                                      # name -> (score, box), best per name
        for (x1, y1, x2, y2), name, score in boxes + people:
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x, d.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = name, score
            d.results.append(hyp)
            out.detections.append(d)
            if score > found.get(name, (0.0,))[0]:
                found[name] = (score, (x1, y1, x2, y2))
        self.det_pub.publish(out)
        annotated = result.plot()                       # BGR with boxes and labels
        if people:
            import cv2
            for (x1, y1, x2, y2), _, score in people:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(annotated, f'PERSON {score:.2f}', (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.img_pub.publish(Image(header=msg.header, height=annotated.shape[0],
                                   width=annotated.shape[1], encoding='bgr8',
                                   step=annotated.shape[1] * 3, data=annotated.tobytes()))
        for name in self.confirmer.update(found, time.monotonic()):
            self.report(name, *found[name], annotated, msg.width)

    def report(self, name, score, box, annotated, width):
        pose = self.robot_pose()
        b = bearing((box[0] + box[2]) / 2, width)
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        snap = f'{time.strftime("%Y%m%d_%H%M%S")}_{name}.jpg'
        import cv2
        cv2.imwrite(os.path.join(self.sightings_dir, snap), annotated)
        event = {'time': stamp, 'animal': name, 'confidence': round(score, 2),
                 'robot': [round(v, 2) for v in pose] if pose else None,
                 'bearing_deg': round(math.degrees(b), 1),
                 'heading_deg': round(math.degrees(pose[2] + b), 1) if pose else None,
                 'snapshot': snap}
        new_file = not os.path.exists(os.path.join(self.sightings_dir, 'sightings.csv'))
        with open(os.path.join(self.sightings_dir, 'sightings.csv'), 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(event))
            if new_file:
                w.writeheader()
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in event.items()})
        self.sight_pub.publish(String(data=json.dumps(event)))
        if name != 'person':                            # intruder.py speaks for people
            self.say_pub.publish(String(data=f'{name.capitalize()} spotted'))
        self.get_logger().info(f'Sighting: {name} ({score:.2f}) at bearing {event["bearing_deg"]} deg')


def main(args=None):
    rclpy.init(args=args)
    node = AnimalDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
