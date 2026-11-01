"""Obstacle rumble: the PS4 controller rumbles when a human driver gets close to something.

In MANUAL and SHARED mode (a person is driving):
  * proximity - like a parking sensor: the free distance ahead of the robot in the
    direction it is driving (from /scan) below `warn_distance` rumbles the controller,
    stronger the closer it gets;
  * bump - a jolt in the IMU's horizontal acceleration while driving at a steady
    command (the robot hit something, e.g. a low rock the lidar cannot see) gives a hard
    0.5 s buzz. Jolts within 0.5 s of a command change are our own starts and stops;
  * contact - pressed against an obstacle, it is closer than the lidar's minimum range
    and disappears from the scan; full rumble is held while still driving into it.
Silent in AUTO (nobody is holding the sticks to feel it), and for a moment after each mode
change so mode_manager's mode pulses come through.
"""

import math
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, JoyFeedback, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from ranger_bot.assist import scan_to_points

ROBOT_FRONT = 0.08        # base_link to the Burger's front/back edge (m)
ROBOT_HALF_WIDTH = 0.09   # half the width incl. wheels (m)


def clearance(points, v, half_width=ROBOT_HALF_WIDTH, front=ROBOT_FRONT, margin=0.03):
    """Free distance (m) in the direction of travel v (sign); inf if nothing is in the way."""
    if abs(v) < 1e-3:
        return math.inf
    best = math.inf
    for x, y in points:
        ahead = x if v > 0 else -x
        if ahead > 0 and abs(y) <= half_width + margin:
            best = min(best, ahead - front)
    return max(0.0, best)


def proximity_intensity(dist, warn=0.4, full=0.05, floor=0.25):
    """0 beyond `warn`, rising linearly from `floor` at `warn` to 1.0 at `full` and closer."""
    if dist >= warn:
        return 0.0
    if dist <= full:
        return 1.0
    return floor + (1.0 - floor) * (warn - dist) / (warn - full)


class Haptics(Node):

    def __init__(self):
        super().__init__('haptics')
        self.warn = self.declare_parameter('warn_distance', 0.4).value
        # Horizontal IMU accel. Measured 2026-10-31: <= 0.05 m/s^2 driving normally,
        # 2.4 m/s^2 driving into a tree at 0.15 m/s.
        self.bump_accel = self.declare_parameter('bump_accel', 1.0).value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lidar_pose = None
        self.points = []
        self.v = 0.0
        self.w = 0.0
        self.cmd_changed = 0.0        # the diff drive has no accel limit: starts and stops
                                      # jolt the IMU as hard as a collision (~2.4 m/s^2)
        self.mode = None
        self.quiet_until = 0.0        # wall time: let mode pulses through
        self.bump_until = 0.0
        self.rumbling = False
        self.contact = False          # obstacle got closer than the lidar's minimum range
        self.last_clearance = math.inf

        self.pub = self.create_publisher(JoyFeedback, 'joy/set_feedback', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'ranger/mode', self.on_mode, latched)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, 'cmd_vel', self.on_cmd, 10)
        self.create_subscription(Imu, 'imu', self.on_imu, qos_profile_sensor_data)
        self.create_timer(0.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_mode(self, msg):
        if msg.data != self.mode:
            self.mode = msg.data
            self.quiet_until = time.monotonic() + 1.2     # longest mode pattern is ~0.9 s
            self.rumbling = False                         # the mode pulses own it now

    def on_scan(self, scan):
        if self.lidar_pose is None:
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, scan.header.frame_id,
                                                    Time(), timeout=Duration(seconds=0.1))
            except TransformException:
                return
            q = t.transform.rotation
            self.lidar_pose = (t.transform.translation.x, t.transform.translation.y,
                               math.atan2(2 * (q.w * q.z + q.x * q.y),
                                          1 - 2 * (q.y * q.y + q.z * q.z)))
        x, y, yaw = self.lidar_pose
        self.points = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                     scan.range_max, x, y, yaw)

    def on_cmd(self, msg):
        v, w = msg.twist.linear.x, msg.twist.angular.z
        if abs(v - self.v) > 0.03 or abs(w - self.w) > 0.2:
            self.cmd_changed = time.monotonic()
        self.v, self.w = v, w

    def on_imu(self, msg):
        a = msg.linear_acceleration
        steady = time.monotonic() - self.cmd_changed > 0.5     # not our own start/stop
        if self.human_driving() and steady and abs(self.v) > 0.02 \
                and math.hypot(a.x, a.y) > self.bump_accel:
            if time.monotonic() > self.bump_until:
                self.get_logger().warn(f'Bump! ({math.hypot(a.x, a.y):.1f} m/s^2)')
            self.bump_until = time.monotonic() + 0.5

    def human_driving(self):
        return self.mode in ('manual', 'shared')

    def tick(self):
        now = time.monotonic()
        if now < self.quiet_until:
            return
        if not self.human_driving():
            intensity = 0.0
        elif now < self.bump_until:
            intensity = 1.0
        else:
            dist = clearance(self.points, self.v)
            # Below range_min (0.12 m) the lidar reports "no return", so an obstacle we are
            # pressed against vanishes. Hold full rumble while still driving towards it.
            if self.v > 0.01 and dist == math.inf and self.last_clearance < 0.1:
                self.contact = True
            if self.contact and (self.v <= 0.01 or dist < math.inf):
                self.contact = False          # backed off / turned away, or seen again
            self.last_clearance = dist if not self.contact else self.last_clearance
            intensity = 1.0 if self.contact else proximity_intensity(dist, warn=self.warn)
        if intensity > 0.0 or self.rumbling:
            # Re-sent at 10 Hz while active; one final 0 switches it off.
            self.pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                         intensity=float(intensity)))
            self.rumbling = intensity > 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Haptics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0, intensity=0.0))
        node.destroy_node()
        rclpy.try_shutdown()
