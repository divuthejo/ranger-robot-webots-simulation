"""Simulated battery: drains while the robot works, sends it home to charge in AUTO.

    /odometry/filtered  (in)  distance driven (EKF)
    /ranger/mode        (in)
    /ranger/patrol_state (in)  patrol.py running/paused/off
    /ranger/battery     (out) sensor_msgs/BatteryState, 1 Hz (percentage 0..1)
    /ranger/task, /ranger/say, /ranger/set_mode (out)

Drain: idle (electronics, lidar, camera) + per metre driven. With the defaults a full battery
lasts ~30 min of patrolling. Charging happens on the dock at home (within `dock_radius` of
0, 0) while standing still.

    <= warn (25 %)   "Battery low" + rumble, once
    <= low (20 %)    AUTO: "Returning to charge" -> task "go towards the home"; MANUAL/SHARED:
                     reminder every 60 s (the human drives, never taken over)
    0 %              "Battery empty" -> stop (mode_manager halt); it can still crawl home in
                     manual (sim kindness), charging works as usual
    charged (95 %)   if this node sent the robot home from a patrol: "patrol" again
/ranger/battery_cmd (String): "drain 15" sets 15 %, "full" - for testing.
"""

import math

IDLE_DRAIN = 0.010          # %/s
DRIVE_DRAIN = 0.30          # % per metre (0.15 m/s patrol: ~30 min on a charge)
CHARGE_RATE = 0.5           # %/s on the dock (a full charge in ~3 min of sim)


class BatteryLogic:
    """ROS-free model + policy. step() -> list of actions."""

    def __init__(self, level=100.0, warn=25.0, low=20.0, charged=95.0, dock_radius=0.45,
                 idle=IDLE_DRAIN, drive=DRIVE_DRAIN, charge=CHARGE_RATE):
        self.level, self.warn, self.low, self.charged = level, warn, low, charged
        self.dock_radius, self.idle, self.drive, self.charge = dock_radius, idle, drive, charge
        self.warned = False
        self.returning = False       # we sent it home
        self.resume_patrol = False
        self.charging = False
        self.empty_sent = False
        self.last_reminder = -1e9
        self.last_charging_said = -1e9

    def on_dock(self, pose):
        return pose is not None and math.hypot(pose[0], pose[1]) <= self.dock_radius

    def step(self, dt, dist, pose, moving, mode, now, patrolling):
        """dt s elapsed, dist m driven in it. Returns actions:
        ('say', text) ('rumble',) ('go_home',) ('patrol',) ('halt',)"""
        acts = []
        docked = self.on_dock(pose) and not moving
        if docked:
            if not self.charging:
                self.charging = True
                # not again when it only turned on the dock (intruder watch) a moment ago
                if self.level < self.charged and now - self.last_charging_said > 60.0:
                    self.last_charging_said = now
                    acts.append(('say', 'Charging'))
            self.level = min(100.0, self.level + self.charge * dt)
        else:
            self.charging = False
            self.level = max(0.0, self.level - self.idle * dt - self.drive * dist)

        if self.level > self.warn + 2:
            self.warned = False
        if self.level > 1.0:
            self.empty_sent = False

        if self.charging:
            if self.level >= self.charged and (self.returning or self.resume_patrol):
                acts.append(('say', 'Battery charged'))
                if self.resume_patrol and mode == 'auto':
                    acts.append(('patrol',))
                self.returning = self.resume_patrol = False
            return acts

        if self.level <= 0.0:
            if not self.empty_sent:
                self.empty_sent = True
                acts += [('say', 'Battery empty'), ('halt',)]
            return acts
        if self.level <= self.low:
            if mode == 'auto' and not self.returning:
                self.returning = True
                self.resume_patrol = patrolling
                acts += [('say', f'Battery at {self.level:.0f} percent. Returning to charge'),
                         ('go_home',), ('rumble',)]
            elif mode != 'auto' and now - self.last_reminder >= 60.0:
                self.last_reminder = now
                acts += [('say', f'Battery at {self.level:.0f} percent. Please drive home'),
                         ('rumble',)]
            if mode != 'auto':
                self.returning = False      # the human took over the trip home
        elif self.level <= self.warn and not self.warned:
            self.warned = True
            acts += [('say', f'Battery low, {self.level:.0f} percent'), ('rumble',)]
        return acts


def main(args=None):
    import time

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.time import Time
    from sensor_msgs.msg import BatteryState, JoyFeedback
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class Battery(Node):
        def __init__(self):
            super().__init__('battery')
            self.logic = BatteryLogic(level=self.declare_parameter('start_level', 100.0).value)
            self.mode, self.patrolling = None, False
            self.dist, self.last_xy, self.speed = 0.0, None, 0.0
            self.tf = Buffer()
            TransformListener(self.tf, self)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Odometry, 'odometry/filtered', self.on_odom, 10)
            self.create_subscription(String, 'ranger/mode', self.on_mode, latched)
            self.create_subscription(String, 'ranger/patrol_state', self.on_patrol, latched)
            self.create_subscription(String, 'ranger/battery_cmd', self.on_cmd, 10)
            self.pub = self.create_publisher(BatteryState, 'ranger/battery', latched)
            self.task_pub = self.create_publisher(String, 'ranger/task', 10)
            self.say_pub = self.create_publisher(String, 'ranger/say', 10)
            self.mode_pub = self.create_publisher(String, 'ranger/set_mode', 10)
            self.rumble_pub = self.create_publisher(JoyFeedback, 'joy/set_feedback', 10)
            self.last = time.monotonic()
            self.rumble_off = None
            self.create_timer(1.0, self.tick)
            self.create_timer(0.1, self.rumble_tick)
            self.get_logger().info(f'Battery at {self.logic.level:.0f} %')

        def on_mode(self, m):
            self.mode = m.data

        def on_patrol(self, m):
            # patrol.py: 'running' / 'paused' / 'off'. When this node sends the robot home the
            # tasks node pauses the patrol, so the value at the moment the battery goes low
            # (before that pause) is what step() records as resume_patrol.
            self.patrolling = m.data == 'running'

        def on_cmd(self, m):
            w = m.data.split()
            if w and w[0] == 'full':
                self.logic.level = 100.0
            elif len(w) == 2 and w[0] == 'drain':
                self.logic.level = max(0.0, min(100.0, float(w[1])))

        def on_odom(self, m):
            p = m.pose.pose.position
            if self.last_xy is not None:
                d = math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
                if d < 1.0:                     # ignore EKF resets / set_pose jumps
                    self.dist += d
            self.last_xy = (p.x, p.y)
            self.speed = abs(m.twist.twist.linear.x) + 0.1 * abs(m.twist.twist.angular.z)

        def pose(self):
            try:
                t = self.tf.lookup_transform('map', 'base_link', Time())
            except TransformException:
                return None
            return t.transform.translation.x, t.transform.translation.y

        def tick(self):
            now = time.monotonic()
            dt, self.last = now - self.last, now
            dist, self.dist = self.dist, 0.0
            acts = self.logic.step(dt, dist, self.pose(), self.speed > 0.02, self.mode, now,
                                   self.patrolling)
            for a in acts:
                if a[0] == 'say':
                    self.say_pub.publish(String(data=a[1]))
                    self.get_logger().info(a[1])
                elif a[0] == 'go_home':
                    self.task_pub.publish(String(data='go towards the home'))
                elif a[0] == 'patrol':
                    self.task_pub.publish(String(data='patrol'))
                elif a[0] == 'halt':
                    self.mode_pub.publish(String(data='stop'))
                elif a[0] == 'rumble':
                    self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                                        intensity=0.6))
                    self.rumble_off = now + 1.2
            b = BatteryState()
            b.header.stamp = self.get_clock().now().to_msg()
            b.percentage = self.logic.level / 100.0
            b.voltage = 10.5 + 2.1 * self.logic.level / 100.0     # 3S Li-ion-ish
            b.present = True
            b.power_supply_status = (BatteryState.POWER_SUPPLY_STATUS_CHARGING
                                     if self.logic.charging else
                                     BatteryState.POWER_SUPPLY_STATUS_DISCHARGING)
            self.pub.publish(b)

        def rumble_tick(self):
            if self.rumble_off and time.monotonic() >= self.rumble_off:
                self.rumble_off = None
                self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                                    intensity=0.0))

    rclpy.init(args=args)
    node = Battery()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
