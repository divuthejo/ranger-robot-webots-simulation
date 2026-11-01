"""Patrol: what AUTO mode does when nobody has given the robot a goal.

    /ranger/mode == 'auto' and nothing else navigating  ->  loop the patrol route
    /ranger/mode leaves 'auto'                           ->  cancel the patrol

The route (parameter `route`, map frame) goes through Nav2's FollowWaypoints action.
The default loop uses the six goals that passed the Nav2 test on 2026-10-31.
Each lap restarts from the waypoint nearest the robot, so re-entering AUTO does not send
it back across the reserve to the first point. Other nodes (the coming Claude mission)
take over by publishing 'pause' on /ranger/patrol, and hand back with 'resume'; a pause
also ends when the robot leaves AUTO.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def parse_route(text):
    """'name:x,y; name:x,y; ...' -> [(name, x, y)]. A string because a ROS parameter list
    cannot mix names and numbers."""
    route = []
    for part in text.split(';'):
        if not part.strip():
            continue
        name, xy = part.split(':')
        x, y = (float(v) for v in xy.split(','))
        route.append((name.strip(), x, y))
    return route


def rotate_to_nearest(route, x, y):
    """The route, starting from the waypoint nearest (x, y) and keeping the loop order."""
    if not route:
        return []
    i = min(range(len(route)), key=lambda k: math.hypot(route[k][1] - x, route[k][2] - y))
    return route[i:] + route[:i]


class Patrol(Node):

    def __init__(self):
        super().__init__('patrol')
        self.route = parse_route(self.declare_parameter(
            'route', 'centre:3,-3; lake:12.5,8; meadow:0,12.5; forest:-9,2.5; '
                     'ridge:2,-9.5; start:0,0').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.mode = None
        self.paused = False
        self.goal_handle = None
        self.sending = False
        self.lap = 0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'ranger/mode', self.on_mode, latched)
        self.create_subscription(String, 'ranger/patrol', self.on_command, 10)
        # "running" (AUTO, not paused) / "paused" (AUTO, paused) / "off" - battery.py resumes
        # the patrol after charging only if it was running
        self.state_pub = self.create_publisher(String, 'ranger/patrol_state', latched)
        self.state = None
        self.create_timer(1.0, self.tick)
        self.get_logger().info(f'Patrol route: {" -> ".join(n for n, _, _ in self.route)} (loop)')

    def on_mode(self, msg):
        if msg.data != self.mode:
            self.mode = msg.data
            if self.mode != 'auto':
                self.cancel('mode is now ' + self.mode)
                # A pause lasts until the robot leaves AUTO, so the next time someone
                # switches to AUTO the patrol runs again. (Not cleared on *entering* auto:
                # a "pause + switch to auto" pair may arrive in either order.)
                self.paused = False

    def on_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'pause':
            self.paused = True
            self.cancel('paused')
        elif cmd == 'resume':
            self.paused = False
            self.get_logger().info('Patrol resumed')
        else:
            self.get_logger().warn(f"Unknown patrol command {cmd!r} (use 'pause' / 'resume')")

    def tick(self):
        state = 'off' if self.mode != 'auto' else 'paused' if self.paused else 'running'
        if state != self.state:
            self.state = state
            self.state_pub.publish(String(data=state))
        if self.mode == 'auto' and not self.paused and self.goal_handle is None \
                and not self.sending:
            self.start_lap()

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time(),
                                                timeout=Duration(seconds=0.2))
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def start_lap(self):
        if not self.client.server_is_ready():
            self.get_logger().info('Waiting for Nav2 (follow_waypoints)...',
                                   throttle_duration_sec=10.0)
            return
        pos = self.robot_xy()
        if pos is None:
            return
        route = rotate_to_nearest(self.route, *pos)
        goal = FollowWaypoints.Goal()
        for _, x, y in route:
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x, p.pose.position.y = x, y
            p.pose.orientation.w = 1.0
            goal.poses.append(p)
        self.lap += 1
        self.sending = True
        self.get_logger().info(f'Patrol lap {self.lap}: {" -> ".join(n for n, _, _ in route)}')
        self.client.send_goal_async(goal).add_done_callback(self.on_accepted)

    def on_accepted(self, future):
        self.sending = False
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Patrol goal rejected')
            return
        if self.mode != 'auto' or self.paused:      # left auto while the goal was in flight
            handle.cancel_goal_async()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_done)

    def on_done(self, future):
        result = future.result().result
        missed = list(result.missed_waypoints)
        if missed:
            self.get_logger().warn(f'Patrol lap {self.lap} done, missed waypoints {missed}')
        else:
            self.get_logger().info(f'Patrol lap {self.lap} done')
        self.goal_handle = None          # tick() starts the next lap while still in auto

    def cancel(self, reason):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
            self.get_logger().info(f'Patrol stopped ({reason})')


def main(args=None):
    rclpy.init(args=args)
    node = Patrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
