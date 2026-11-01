"""Give AMCL a tight initial pose, then exit.

    ros2 run ranger_bot initial_pose --ros-args -p x:=0.0 -p y:=0.0 -p yaw:=0.0

AMCL's set_initial_pose only takes x/y/yaw and starts with a wide particle spread
(~0.5 m). In open grass the lidar sees almost nothing to shrink that spread, so the
particle cloud split into clusters and the estimate jumped between them - 2.3 m off on
a 4 m drive across the meadow, while the EKF odometry alone was within 0.1 m. The robot
always spawns at a precisely known pose, so publish it with a small covariance instead.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


class InitialPose(Node):

    def __init__(self):
        super().__init__('initial_pose')
        self.x = self.declare_parameter('x', 0.0).value
        self.y = self.declare_parameter('y', 0.0).value
        self.yaw = self.declare_parameter('yaw', 0.0).value
        self.xy_std = self.declare_parameter('xy_std', 0.05).value
        self.yaw_std = self.declare_parameter('yaw_std', 0.03).value
        self.pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.sent = 0
        self.done = False
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        # Wait for AMCL to subscribe and for sim time to start, then send a few times
        # (a single message can be lost while AMCL is still activating).
        if self.pub.get_subscription_count() == 0 or self.get_clock().now().nanoseconds == 0:
            self.get_logger().info('Waiting for AMCL...', throttle_duration_sec=5.0)
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2)
        msg.pose.covariance[0] = self.xy_std ** 2
        msg.pose.covariance[7] = self.xy_std ** 2
        msg.pose.covariance[35] = self.yaw_std ** 2
        self.pub.publish(msg)
        self.sent += 1
        if self.sent >= 3:
            self.get_logger().info(f'Initial pose ({self.x}, {self.y}, {self.yaw}) sent to AMCL '
                                   f'(std {self.xy_std} m, {self.yaw_std} rad).')
            self.timer.cancel()
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = InitialPose()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
