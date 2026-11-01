"""Startup guard: make sure every Nav2 lifecycle node really becomes ACTIVE.

Nav2's lifecycle managers give up when one change_state service call times out (they have no
service-timeout parameter). Under startup load that happened (2026-11-01): map_server, AMCL
and bt_navigator stayed inactive -> no map frame ("Cannot plan: robot position unknown") and
Nav2 rejected every goal. This node checks the nodes every `period` s for `duration` s and
configures/activates any that are stuck, with generous timeouts. If it had to activate AMCL it
re-sends the starting pose (AMCL ignores /initialpose while inactive). Exits once all nodes
were active on `settle` consecutive checks.
"""

import math
import time

NAV2_NODES = ['map_server', 'amcl', 'rock_mask_server', 'controller_server', 'smoother_server',
              'planner_server', 'route_server', 'behavior_server', 'velocity_smoother',
              'collision_monitor', 'bt_navigator', 'waypoint_follower', 'docking_server']

# lifecycle_msgs/Transition ids
CONFIGURE, ACTIVATE = 1, 3


def transitions_for(state):
    """Lifecycle state label -> transitions that bring it to 'active' ([] if none/unknown)."""
    return {'unconfigured': [CONFIGURE, ACTIVATE], 'inactive': [ACTIVATE]}.get(state, [])


def main(args=None):
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from lifecycle_msgs.srv import ChangeState, GetState
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node('lifecycle_guard')
    nodes = node.declare_parameter('nodes', NAV2_NODES).value
    period = node.declare_parameter('period', 10.0).value
    duration = node.declare_parameter('duration', 300.0).value
    settle = node.declare_parameter('settle', 3).value
    start = node.declare_parameter('initial_pose', [0.0, 0.0, 0.0]).value
    log = node.get_logger()
    clients = {n: (node.create_client(GetState, f'/{n}/get_state'),
                   node.create_client(ChangeState, f'/{n}/change_state')) for n in nodes}
    pose_pub = node.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

    def call(cli, req, timeout):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def send_initial_pose():
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.header.stamp = node.get_clock().now().to_msg()
        x, y, yaw = start
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.05 ** 2
        m.pose.covariance[35] = 0.03 ** 2
        for _ in range(3):
            pose_pub.publish(m)
            rclpy.spin_once(node, timeout_sec=0.3)

    t0, ok_streak = time.time(), 0
    try:
        while time.time() - t0 < duration and ok_streak < settle:
            all_ok = True
            for name, (gs, cs) in clients.items():
                r = call(gs, GetState.Request(), 10.0)
                if r is None:
                    all_ok = False                  # not up yet (or not reachable): next round
                    continue
                state = r.current_state.label
                if state == 'active':
                    continue
                all_ok = False
                steps = transitions_for(state)
                for tid in steps:
                    q = ChangeState.Request()
                    q.transition.id = tid
                    res = call(cs, q, 30.0)
                    if not (res and res.success):
                        log.warn(f'{name}: transition {tid} from {state} failed, will retry')
                        break
                else:
                    if steps:
                        log.warn(f'{name} was stuck {state} - activated it')
                        if name == 'amcl':
                            send_initial_pose()
                            log.info('re-sent the initial pose to AMCL')
            ok_streak = ok_streak + 1 if all_ok else 0
            end = time.time() + period
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.2)
        log.info('all Nav2 nodes active' if ok_streak >= settle else
                 'gave up waiting for some Nav2 nodes')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
