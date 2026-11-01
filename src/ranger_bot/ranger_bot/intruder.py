"""Intruder alert: a person in the reserve -> alarm, rumble, log; in AUTO the robot watches them.

    /ranger/animals   (in)  tracks from animal_tracker.py ('person' class = intruder)
    /ranger/mode      (in)  drive mode (only AUTO turns the robot)
    /ranger/say, /joy/set_feedback, /ranger/patrol (out)
    /ranger/intruder  (out) std_msgs/String JSON: {"state": "detected"|"tracking"|"lost", x, y}
    sightings/intruders.csv

In AUTO the patrol is paused and the robot turns to face the intruder (Nav2 Spin), keeping
them in camera view while they are tracked; once they are out of sight for `lost_after` s the
patrol resumes (only if this node paused it). In MANUAL/SHARED it only alerts - the human
drives. The detector saves the photo (sightings/) like any sighting.
"""

import csv
import json
import math
import os
import time

from ranger_bot.tasks import PLACES

ALARM = [(1.0, 0.4), (0.0, 0.15), (1.0, 0.4), (0.0, 0.15), (1.0, 0.4), (0.0, 0.05)]


def nearest_place(x, y):
    name, (px, py, _) = min(PLACES.items(), key=lambda kv: math.hypot(kv[1][0] - x,
                                                                      kv[1][1] - y))
    return name


class IntruderLogic:
    """State machine, ROS-free. feed() each tracker update -> list of actions.

    actions: ('alert', x, y) ('face', yaw_error) ('lost',) ('resume_patrol',)
    """

    def __init__(self, fresh=3.0, lost_after=20.0, face_tol=math.radians(20)):
        self.fresh, self.lost_after, self.face_tol = fresh, lost_after, face_tol
        self.active = False
        self.last_seen = None
        self.paused_patrol = False
        self.pos = None

    def feed(self, tracks, robot_pose, mode, now):
        acts = []
        people = [t for t in tracks.values() if t.get('cls') == 'person' and
                  t.get('age_s', 1e9) <= self.fresh]
        if people:
            p = min(people, key=lambda t: t['age_s'])
            self.pos, self.last_seen = (p['x'], p['y']), now - p['age_s']
            if not self.active:
                self.active = True
                acts.append(('alert', p['x'], p['y']))
                if mode == 'auto':
                    self.paused_patrol = True
            if mode == 'auto' and robot_pose:
                x, y, yaw = robot_pose
                err = math.atan2(p['y'] - y, p['x'] - x) - yaw
                err = math.atan2(math.sin(err), math.cos(err))
                if abs(err) > self.face_tol:
                    acts.append(('face', err))
        elif self.active and now - self.last_seen > self.lost_after:
            self.active = False
            acts.append(('lost',))
            if self.paused_patrol and mode == 'auto':
                acts.append(('resume_patrol',))
            self.paused_patrol = False
        if mode != 'auto':
            self.paused_patrol = False          # the human took over; never resume behind them
        return acts


def main(args=None):
    import rclpy
    from nav2_msgs.action import Spin
    from rclpy.action import ActionClient
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.time import Time
    from sensor_msgs.msg import JoyFeedback
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class Intruder(Node):
        def __init__(self):
            super().__init__('intruder')
            self.dir = os.path.expanduser(self.declare_parameter(
                'sightings_dir', '~/ranger_ws/sightings').value)
            self.logic = IntruderLogic(
                lost_after=self.declare_parameter('lost_after', 20.0).value)
            self.mode = None
            self.tf = Buffer()
            TransformListener(self.tf, self)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, 'ranger/animals', self.on_tracks, latched)
            self.create_subscription(String, 'ranger/mode',
                                     lambda m: setattr(self, 'mode', m.data), latched)
            self.say_pub = self.create_publisher(String, 'ranger/say', 10)
            self.patrol_pub = self.create_publisher(String, 'ranger/patrol', 10)
            self.state_pub = self.create_publisher(String, 'ranger/intruder', 10)
            self.rumble_pub = self.create_publisher(JoyFeedback, 'joy/set_feedback', 10)
            self.spin = ActionClient(self, Spin, 'spin')
            self.spinning = False
            self.rumble_steps, self.rumble_next = [], 0.0
            self.create_timer(0.05, self.rumble_tick)
            self.get_logger().info('Intruder alert ready')

        def pose(self):
            try:
                t = self.tf.lookup_transform('map', 'base_link', Time())
            except TransformException:
                return None
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw

        def on_tracks(self, msg):
            try:
                tracks = json.loads(msg.data)
            except ValueError:
                return
            for act in self.logic.feed(tracks, self.pose(), self.mode, time.time()):
                getattr(self, 'do_' + act[0])(*act[1:])
            if self.logic.active and self.logic.pos:
                self.state(('tracking',), *self.logic.pos)

        def state(self, st, x=None, y=None):
            self.state_pub.publish(String(data=json.dumps(
                {'state': st[0], 'x': x, 'y': y, 'time': time.strftime('%H:%M:%S')})))

        def do_alert(self, x, y):
            place = nearest_place(x, y)
            self.say_pub.publish(String(data=f'Warning. Intruder detected near the {place}'))
            self.rumble_steps, self.rumble_next = list(ALARM), 0.0
            if self.mode == 'auto':
                self.patrol_pub.publish(String(data='pause'))
            self.state(('detected',), x, y)
            path = os.path.join(self.dir, 'intruders.csv')
            new = not os.path.exists(path)
            os.makedirs(self.dir, exist_ok=True)
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['time', 'x', 'y', 'near', 'robot', 'mode'])
                p = self.pose()
                w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), round(x, 2), round(y, 2), place,
                            json.dumps([round(v, 2) for v in p]) if p else '', self.mode])
            self.get_logger().warn(f'INTRUDER at ({x:.1f}, {y:.1f}) near the {place}')

        def do_face(self, err):
            if self.spinning or not self.spin.server_is_ready():
                return
            goal = Spin.Goal()
            goal.target_yaw = float(err)
            goal.time_allowance.sec = 15
            self.spinning = True
            fut = self.spin.send_goal_async(goal)

            def accepted(f):
                h = f.result()
                if not h.accepted:
                    self.spinning = False
                    return
                h.get_result_async().add_done_callback(
                    lambda _: setattr(self, 'spinning', False))
            fut.add_done_callback(accepted)

        def do_lost(self):
            self.say_pub.publish(String(data='Intruder out of sight'))
            self.state(('lost',), *(self.logic.pos or (None, None)))
            self.get_logger().info('Intruder lost')

        def do_resume_patrol(self):
            self.patrol_pub.publish(String(data='resume'))

        def rumble_tick(self):
            now = time.monotonic()
            if not self.rumble_steps or now < self.rumble_next:
                return
            intensity, duration = self.rumble_steps.pop(0)
            self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                                intensity=float(intensity)))
            self.rumble_next = now + duration

    rclpy.init(args=args)
    node = Intruder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
