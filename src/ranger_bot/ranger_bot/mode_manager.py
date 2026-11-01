"""Mode manager: the one node that drives /cmd_vel, switching between three modes live.

    manual  /cmd_vel_joy     PS4 joystick straight through (teleop_twist_joy)
    shared  /cmd_vel_shared  joystick + collision avoidance (shared_control / assist.py)
    auto    /cmd_vel_auto    Nav2 (collision_monitor output) or mapping_tour

Switch modes with the PS4 face buttons - Square: manual, Circle: shared, Triangle: auto -
or by publishing 'manual' / 'shared' / 'auto' on /ranger/set_mode (std_msgs/String).
The current mode is published (latched) on /ranger/mode.

Safety:
  * Joystick takeover: in auto the sticks are ignored, but holding L1/R1 with a stick
    pushed for `takeover_hold` seconds (1 s) switches to shared (the driver gets control,
    collision avoidance stays on) and cancels Nav2 goals. The hold stops a brushed stick
    from cancelling a mission.
  * A stop is sent on every mode switch and whenever the selected source goes quiet.
  * Halt: 'stop' on /ranger/set_mode (the voice/CLI "stop" task) stops the robot in any
    mode and ignores every velocity source - even a stick still held - until L1/R1 is
    released and pressed again, or a mode is chosen. From auto it also cancels Nav2 and
    drops to manual (else the patrol would just restart).

Speed: every forwarded command (all modes) is scaled by a speed level - 25/50/75/100 %,
default 100 % - so a slow level also slows Nav2 (same path: linear and angular scale
together). D-pad up/down steps it; /ranger/speed takes 'faster' | 'slower' | 'max' | 'min'
(voice: "speed up", "slow down", "max speed", "minimum speed"). Announced on /ranger/say.

Feedback: the controller rumbles on every mode change - 1 pulse manual, 2 shared,
3 auto, one long buzz for a joystick takeover or a halt - via joy_node's /joy/set_feedback.
"""

import time

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import TwistStamped
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Joy, JoyFeedback
from std_msgs.msg import String

MODES = ('manual', 'shared', 'auto')
SOURCES = {'manual': 'cmd_vel_joy', 'shared': 'cmd_vel_shared', 'auto': 'cmd_vel_auto'}

# PS4 layout on this laptop (see config/ps4_teleop.yaml).
BUTTON_MODES = {3: 'manual', 1: 'shared', 2: 'auto'}   # Square, Circle, Triangle
DEADMAN_BUTTONS = (4, 5)                               # L1, R1
STICK_AXES = (1, 3)                                    # left-Y (drive), right-X (turn)

# Rumble patterns: list of (intensity 0..1, seconds). 1/2/3 pulses = manual/shared/auto.
PULSE = [(0.7, 0.15), (0.0, 0.15)]
RUMBLE = {'manual': PULSE, 'shared': PULSE * 2, 'auto': PULSE * 3,
          'takeover': [(1.0, 0.8), (0.0, 0.1)], 'halt': [(1.0, 0.8), (0.0, 0.1)]}
HALT_WORDS = ('stop', 'halt')
DPAD_Y_AXIS = 7                                        # D-pad up/down (+1 = up)


class Speed:
    """Speed level as a fraction of full speed; D-pad and word commands step it."""

    LEVELS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, index=len(LEVELS) - 1):
        self.index = index
        self.prev_dpad = 0

    @property
    def fraction(self):
        return self.LEVELS[self.index]

    def command(self, word):
        """'faster' | 'slower' | 'max' | 'min' -> True if the level changed."""
        target = {'faster': self.index + 1, 'slower': self.index - 1,
                  'max': len(self.LEVELS) - 1, 'min': 0}.get(word.strip().lower())
        if target is None:
            raise ValueError(f'unknown speed command {word!r}')
        target = max(0, min(len(self.LEVELS) - 1, target))
        changed, self.index = target != self.index, target
        return changed

    def on_dpad(self, value):
        """D-pad Y axis value (+1 up) -> 'faster' / 'slower' on a press, else None."""
        pos = 1 if value > 0.5 else -1 if value < -0.5 else 0
        pressed = pos if pos != self.prev_dpad else 0
        self.prev_dpad = pos
        return {1: 'faster', -1: 'slower'}.get(pressed)


class ModeLogic:
    """ROS-free mode selection, so it can be unit-tested."""

    def __init__(self, mode='manual', takeover_threshold=0.2, takeover_hold=1.0):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}, expected one of {MODES}')
        self.mode = mode
        self.takeover_threshold = takeover_threshold
        self.takeover_hold = takeover_hold
        self.takeover_since = None      # when L1/R1 + stick started being held in auto
        self.prev_buttons = []
        self.halted = False             # 'stop': no velocity passes until L1/R1 is re-pressed
        self.deadman_held = False

    def halt(self):
        """Stop in place; returns True if this left auto (Nav2 goals must be cancelled)."""
        self.halted = True
        self.takeover_since = None
        if self.mode == 'auto':
            self.mode = 'manual'
            return True
        return False

    def passes(self, topic):
        """Whether a velocity command on `topic` should reach the robot."""
        return topic == SOURCES[self.mode] and not self.halted

    def set_mode(self, mode):
        """Switch mode; returns True if it changed. Unknown modes raise ValueError."""
        mode = mode.strip().lower()
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}, expected one of {MODES}')
        changed = mode != self.mode or self.halted
        self.mode = mode
        self.takeover_since = None
        self.halted = False
        return changed

    def on_joy(self, buttons, axes, now):
        """Handle a joystick message at time `now` (s); returns (new_mode or None, reason)."""
        pressed = [i for i, b in enumerate(buttons)
                   if b and (i >= len(self.prev_buttons) or not self.prev_buttons[i])]
        self.prev_buttons = list(buttons)
        deadman = any(i < len(buttons) and buttons[i] for i in DEADMAN_BUTTONS)
        fresh_deadman, self.deadman_held = deadman and not self.deadman_held, deadman
        for i in pressed:                       # rising edges only: holding is one press
            if i in BUTTON_MODES and (BUTTON_MODES[i] != self.mode or self.halted):
                self.mode = BUTTON_MODES[i]
                self.takeover_since = None
                self.halted = False
                return self.mode, 'button'
        if self.halted:
            if fresh_deadman and self.mode != 'auto':
                self.halted = False
                return None, 'resume'
            return None, ''
        if self.mode != 'auto':
            self.takeover_since = None
            return None, ''
        stick = max((abs(axes[i]) for i in STICK_AXES if i < len(axes)), default=0.0)
        if not (deadman and stick > self.takeover_threshold):
            self.takeover_since = None          # let go: the hold starts over
            return None, ''
        if self.takeover_since is None:
            self.takeover_since = now
        if now - self.takeover_since >= self.takeover_hold:
            self.mode = 'shared'
            self.takeover_since = None
            return self.mode, 'takeover'
        return None, ''

    def source(self):
        return SOURCES[self.mode]


class ModeManager(Node):

    def __init__(self):
        super().__init__('mode_manager')
        self.logic = ModeLogic(self.declare_parameter('initial_mode', 'manual').value,
                               takeover_hold=self.declare_parameter('takeover_hold', 1.0).value)
        self.timeout = self.declare_parameter('source_timeout', 0.5).value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.speed = Speed()
        self.dpad_axis = self.declare_parameter('dpad_axis', DPAD_Y_AXIS).value
        self.dpad_up_sign = float(self.declare_parameter('dpad_up_sign', 1.0).value)

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, 'ranger/mode', latched)
        self.last_msg = {topic: None for topic in SOURCES.values()}
        self.stopped = True     # whether a stop has been sent since the source went quiet
        for topic in SOURCES.values():
            self.create_subscription(TwistStamped, topic,
                                     lambda m, t=topic: self.on_cmd(t, m), 10)
        self.create_subscription(Joy, 'joy', self.on_joy, 10)
        self.create_subscription(String, 'ranger/set_mode', self.on_set_mode, 10)
        self.create_subscription(String, 'ranger/speed', lambda m: self.change_speed(m.data), 10)
        self.cancel_clients = [
            self.create_client(CancelGoal, f'{action}/_action/cancel_goal')
            for action in ('follow_waypoints', 'navigate_to_pose', 'navigate_through_poses')]
        self.create_timer(0.1, self.watchdog)
        # Rumble runs on the wall clock so a slow or paused simulation does not stretch it.
        self.rumble_pub = self.create_publisher(JoyFeedback, 'joy/set_feedback', 10)
        self.say_pub = self.create_publisher(String, 'ranger/say', 10)     # announcer.py
        self.rumble_steps = []
        self.rumble_next = 0.0
        self.create_timer(0.05, self.rumble_tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.announce('start')

    def announce(self, reason):
        self.mode_pub.publish(String(data=self.logic.mode))
        self.get_logger().info(f'Mode: {self.logic.mode.upper()} ({reason}) - '
                               f'driving from /{self.logic.source()}')

    def switch(self, reason, leaving_auto):
        self.stop()
        if leaving_auto:
            self.cancel_navigation()
        self.announce(reason)
        self.rumble('takeover' if reason == 'takeover' else self.logic.mode)

    def rumble(self, pattern):
        self.rumble_steps = list(RUMBLE[pattern])
        self.rumble_next = 0.0

    def rumble_tick(self):
        now = time.monotonic()
        if not self.rumble_steps or now < self.rumble_next:
            return
        intensity, duration = self.rumble_steps.pop(0)
        self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                            intensity=float(intensity)))
        self.rumble_next = now + duration

    def change_speed(self, word):
        try:
            changed = self.speed.command(word)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        pct = round(self.speed.fraction * 100)
        self.get_logger().info(f'Speed {pct} %' + ('' if changed else ' (unchanged)'))
        if changed:
            self.say_pub.publish(String(data=f'Speed {pct} percent'))
        else:
            edge = 'Full speed' if self.speed.fraction == 1.0 else 'Slowest speed'
            self.say_pub.publish(String(data=f'Already at {edge.lower()}'))

    def on_joy(self, msg):
        if len(msg.axes) > self.dpad_axis:
            step = self.speed.on_dpad(msg.axes[self.dpad_axis] * self.dpad_up_sign)
            if step:
                self.change_speed(step)
        before = self.logic.mode
        mode, reason = self.logic.on_joy(msg.buttons, msg.axes, time.monotonic())
        if mode is not None:
            if reason == 'takeover':
                self.get_logger().warn('Joystick takeover (L1/R1 + stick held) - leaving AUTO')
                self.say_pub.publish(String(data='Joystick takeover'))
            self.switch(reason, before == 'auto')
        elif reason == 'resume':
            self.get_logger().info(f'Resumed ({self.logic.mode.upper()}) - L1/R1 pressed again')
            self.rumble(self.logic.mode)
            self.say_pub.publish(String(data='Resumed'))

    def on_set_mode(self, msg):
        before = self.logic.mode
        if msg.data.strip().lower() in HALT_WORDS:
            left_auto = self.logic.halt()
            self.stop()
            if left_auto:
                self.cancel_navigation()
            self.announce('halt')
            self.get_logger().warn('HALTED - release and press L1/R1 again (or pick a mode) to drive')
            self.rumble('halt')
            return
        try:
            changed = self.logic.set_mode(msg.data)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        if changed:
            self.switch('request', before == 'auto')

    def on_cmd(self, topic, msg):
        self.last_msg[topic] = self.get_clock().now()
        if self.logic.passes(topic):
            f = self.speed.fraction
            if f != 1.0:
                msg.twist.linear.x *= f
                msg.twist.linear.y *= f
                msg.twist.angular.z *= f
            self.cmd_pub.publish(msg)
            self.stopped = False

    def watchdog(self):
        last = self.last_msg[self.logic.source()]
        quiet = last is None or (self.get_clock().now() - last).nanoseconds * 1e-9 > self.timeout
        if quiet and not self.stopped:
            self.stop()

    def stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        self.cmd_pub.publish(msg)
        self.stopped = True

    def cancel_navigation(self):
        # An empty CancelGoal request (zero goal id and stamp) cancels every goal. The
        # patrol's waypoint follower goes first, or it would just start the next waypoint.
        for client in self.cancel_clients:
            if client.service_is_ready():
                client.call_async(CancelGoal.Request())
        self.get_logger().info('Cancelled Nav2 goals')


def main(args=None):
    rclpy.init(args=args)
    node = ModeManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():                  # after a SIGTERM the context is gone: can't publish
            node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
