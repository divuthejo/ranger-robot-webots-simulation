"""Shared-control node: the joystick driver steers, the robot keeps them from crashing.

    /cmd_vel_joy  (TwistStamped, from teleop_twist_joy)  --+
    /scan         (LaserScan)                              +--> /cmd_vel_shared (TwistStamped)
                                                               /shared_control/intervention (Float32, 0..1)

See assist.py for the algorithm.
"""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener

from ranger_bot.assist import AssistParams, correct, scan_to_points


class SharedControl(Node):

    def __init__(self):
        super().__init__('shared_control')
        defaults = AssistParams()
        self.params = AssistParams(
            robot_radius=self.declare_parameter('robot_radius', defaults.robot_radius).value,
            margin=self.declare_parameter('margin', defaults.margin).value,
            horizon=self.declare_parameter('horizon', defaults.horizon).value,
        )
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.scan_timeout = self.declare_parameter('scan_timeout', 0.5).value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lidar_pose = None      # (x, y, yaw) of the scan frame in base_frame
        self.obstacles = None
        self.last_scan_time = None

        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, 'cmd_vel_joy', self.on_cmd, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel_shared', 10)
        self.intervention_pub = self.create_publisher(Float32, 'shared_control/intervention', 10)

    def on_scan(self, scan):
        if self.lidar_pose is None:
            self.lidar_pose = self.lookup_lidar_pose(scan.header.frame_id)
            if self.lidar_pose is None:
                return
        x, y, yaw = self.lidar_pose
        self.obstacles = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                        scan.range_max, x, y, yaw)
        self.last_scan_time = self.get_clock().now()

    def lookup_lidar_pose(self, frame):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(f'Waiting for {self.base_frame} -> {frame} transform: {e}',
                                   throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def on_cmd(self, msg):
        v, w = msg.twist.linear.x, msg.twist.angular.z
        stale = (self.last_scan_time is None or
                 (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9 > self.scan_timeout)
        if stale:
            # No fresh lidar: we cannot vouch for anything but turning in place.
            self.get_logger().warn('No recent /scan, only rotation allowed', throttle_duration_sec=2.0)
            v_out, w_out, intervention = 0.0, w, (1.0 if abs(v) > 1e-3 else 0.0)
        else:
            v_out, w_out, intervention = correct(v, w, self.obstacles, self.params)

        out = TwistStamped()
        out.header = msg.header
        out.twist.linear.x = v_out
        out.twist.angular.z = w_out
        self.cmd_pub.publish(out)
        self.intervention_pub.publish(Float32(data=intervention))


def main(args=None):
    rclpy.init(args=args)
    node = SharedControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
