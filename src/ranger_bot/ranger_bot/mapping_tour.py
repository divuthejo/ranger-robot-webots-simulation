"""Drive a fixed tour of the reserve so slam_toolbox can build the map without a joystick.

    ros2 launch ranger_bot mapping.launch.py teleop:=none
    ros2 run ranger_bot mapping_tour

A simple go-to-point controller heads for each waypoint, and its command goes
through the same collision-avoidance correction as shared control (assist.py),
so the robot steers around trees and rocks that lie on the straight line.
The pose is the simulator's ground truth (/gps + /imu yaw), so the tour follows the
route even if SLAM gets confused; SLAM itself only sees /scan and odometry. This is a
simulation-only mapping helper - the robot's own navigation uses the SLAM map.
"""

import math
import os

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu, LaserScan
from slam_toolbox.srv import SaveMap
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from ranger_bot import keepout_mask
from ranger_bot.assist import AssistParams, correct, scan_to_points

# Clockwise through forest -> meadow -> lake -> rocky ridge, a loop through the centre,
# then back to the start to close the loop. Points sit between the objects in nature_reserve.wbt.
TOUR = [
    ('forest edge', -5.0, 0.0),
    ('forest south', -9.5, -6.0),
    ('forest west', -13.0, -2.0),
    ('forest north-west', -13.0, 2.5),   # around pine 2, which sits on the direct line
    ('forest middle', -9.0, 2.5),
    ('forest north', -7.0, 7.0),
    ('meadow west', -6.0, 12.0),
    ('meadow north', 1.0, 13.8),
    ('meadow east', 6.0, 13.0),
    ('lake north-east', 13.0, 11.0),
    ('lake east', 13.5, 3.0),
    ('lake south', 6.0, 1.5),
    ('ridge east', 11.0, -7.0),
    ('ridge middle', 5.0, -7.0),
    ('ridge west', -2.0, -6.0),
    ('ridge far west', -8.0, -12.0),
    # Loop through the centre, which the perimeter route above leaves unmapped
    # (the LDS-01 only reaches ~3.5 m). Resume here with -p start:=17.
    ('centre north-west', -3.0, 4.0),
    ('centre north', 1.0, 6.0),
    ('centre east', 3.5, 2.0),
    ('centre south-east', 6.0, -3.0),
    ('centre south', 1.5, -4.0),
    ('centre south-west', -3.5, -2.5),
    ('start', 0.0, 0.0),
]


class MappingTour(Node):

    def __init__(self):
        super().__init__('mapping_tour')
        self.speed = self.declare_parameter('speed', 0.2).value
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.4).value
        self.stuck_timeout = self.declare_parameter('stuck_timeout', 30.0).value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        # 1-based waypoint to start from, to resume an interrupted tour.
        start = self.declare_parameter('start', 1).value
        # Where to save the map when the tour ends ('' = don't save).
        self.map_path = os.path.expanduser(self.declare_parameter(
            'map_path', '~/ranger_ws/src/ranger_bot/maps/reserve').value)
        self.save_client = self.create_client(SaveMap, 'slam_toolbox/save_map')
        self.done = False   # set once the tour is over and the map save has finished

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.params = AssistParams()
        self.obstacles = None
        self.lidar_pose = None
        self.position = None
        self.yaw = None

        self.index = max(0, min(len(TOUR) - 1, start - 1))
        self.best_dist = math.inf
        self.last_progress = None

        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PointStamped, 'gps', self.on_gps, 10)
        self.create_subscription(Imu, 'imu', self.on_imu, 10)
        # Auto-mode input of mode_manager (which owns /cmd_vel).
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel_auto', 10)
        self.timer = self.create_timer(0.1, self.step)
        self.get_logger().info(f'Mapping tour: {len(TOUR)} waypoints, starting at {self.index + 1}')

    def lookup(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time(),
                                                timeout=Duration(seconds=0.05))
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def on_gps(self, msg):
        self.position = (msg.point.x, msg.point.y)

    def on_imu(self, msg):
        q = msg.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def on_scan(self, scan):
        if self.lidar_pose is None:
            self.lidar_pose = self.lookup(self.base_frame, scan.header.frame_id)
            if self.lidar_pose is None:
                return
        x, y, yaw = self.lidar_pose
        self.obstacles = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                        scan.range_max, x, y, yaw)

    def publish(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self.cmd_pub.publish(msg)

    def next_waypoint(self, reason):
        name = TOUR[self.index][0]
        self.get_logger().info(f'[{self.index + 1}/{len(TOUR)}] {name}: {reason}')
        self.index += 1
        self.best_dist = math.inf
        self.last_progress = None
        if self.index == len(TOUR):
            self.publish(0.0, 0.0)
            self.timer.cancel()
            self.save_map()

    def save_map(self):
        if not self.map_path:
            self.get_logger().info('Tour finished - save the map now.')
            self.done = True
            return
        if not self.save_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Tour finished but slam_toolbox/save_map is unavailable - '
                                    'save the map manually.')
            self.done = True
            return
        os.makedirs(os.path.dirname(self.map_path), exist_ok=True)
        req = SaveMap.Request(name=String(data=self.map_path))
        self.save_client.call_async(req).add_done_callback(self.on_map_saved)
        self.get_logger().info(f'Tour finished - saving map to {self.map_path}...')

    def on_map_saved(self, future):
        result = future.result()
        if result is not None and result.result == SaveMap.Response.RESULT_SUCCESS:
            self.get_logger().info(f'Map saved: {self.map_path}.yaml/.pgm')
            # The rock mask must share the new map's grid (see keepout_mask.py).
            try:
                keepout_mask.main(['--map', self.map_path + '.yaml'])
            except Exception as e:    # never lose the tour over the mask
                self.get_logger().error(f'Rock mask not regenerated ({e}) - run keepout_mask.')
        else:
            code = None if result is None else result.result
            self.get_logger().error(f'Map save failed (result {code}) - save it manually.')
        self.done = True

    def step(self):
        if self.position is None or self.yaw is None or self.obstacles is None:
            self.get_logger().info('Waiting for /gps, /imu and /scan...', throttle_duration_sec=5.0)
            return
        (x, y), yaw = self.position, self.yaw
        name, gx, gy = TOUR[self.index]
        dist = math.hypot(gx - x, gy - y)
        now = self.get_clock().now()

        if dist < self.goal_tolerance:
            self.next_waypoint('reached')
            return
        if dist < self.best_dist - 0.1:
            self.best_dist = dist
            self.last_progress = now
        elif self.last_progress is None:
            self.last_progress = now
        elif (now - self.last_progress).nanoseconds * 1e-9 > self.stuck_timeout:
            nearest = min((math.hypot(px, py) for px, py in self.obstacles), default=math.inf)
            self.next_waypoint(f'skipped, no progress for {self.stuck_timeout:.0f} s '
                               f'(at {x:.1f}, {y:.1f}, nearest obstacle {nearest:.2f} m)')
            return

        # Turn towards the goal, slowing down while the heading error is large.
        err = math.atan2(gy - y, gx - x) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        w = max(-1.5, min(1.5, 2.0 * err))
        v = self.speed * max(0.0, math.cos(err)) ** 2
        v, w, _ = correct(v, w, self.obstacles, self.params)
        self.publish(v, w)


def main(args=None):
    rclpy.init(args=args)
    node = MappingTour()
    try:
        # Exit when the tour is over, so a finished tour doesn't linger (and spam TF
        # errors) after the simulation is restarted.
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()
