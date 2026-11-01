"""Tasks in AUTO mode: "go towards the deer", "go near the pond", "patrol", "stop".

    /ranger/task (std_msgs/String, plain words)  ->  Nav2 goal  ->  /ranger/task_status

Send one from a terminal:   ros2 run ranger_bot task "go towards the deer"
Voice control and the Claude mission will publish the same plain text.

Targets are named places plus the animals in the world file (positions read from the
.wbt, so moving an animal in Webots only needs a restart). You cannot drive *into* a deer
or a pond, so for an object the goal is a standing spot `standoff` metres from its edge,
on the side facing the robot, turned to face it; if that spot is not free in the global
costmap, other spots around the object are tried, nearest to the robot first.

A task pauses the patrol and switches to AUTO. At a place the robot then waits there; at an
animal it FOLLOWS it ("follow the deer" says the same): whenever the tracked animal has moved
more than `follow_move` m it walks to a new standing spot (quietly - no "Going to" every few
seconds), until the animal is out of sight for `follow_lost` s, `follow_time` s have passed, or
any new task / "stop" / leaving AUTO.
"patrol" resumes the patrol, "stop" switches to MANUAL (which stops and cancels).
"""

import json
import math
import os
import re
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

# name: (x, y, radius of the thing itself). Map frame == Webots world frame.
PLACES = {
    'lake': (8.0, 7.0, 3.5),
    'meadow': (0.0, 12.5, 0.0),
    'forest': (-9.0, 2.5, 0.0),
    'ridge': (2.0, -9.5, 0.0),
    'centre': (3.0, -3.0, 0.0),
    'home': (0.0, 0.0, 0.0),
}
ALIASES = {'pond': 'lake', 'water': 'lake', 'center': 'centre', 'start': 'home', 'base': 'home'}
ANIMAL_RADIUS = {'deer': 0.6, 'horse': 0.9, 'cow': 0.9, 'sheep': 0.5,
                 'dog': 0.4, 'fox': 0.35, 'rabbit': 0.2, 'cat': 0.25}


def target_position(name, targets, seen, max_age=600.0):
    """(x, y, radius) to go to, and a note. Animals wander: use where the tracker last saw it
    (/ranger/animals) if recent, else the spot it started at in the world file."""
    base = targets[name]
    if name not in ANIMAL_RADIUS:
        return base, ''
    t = seen.get(name)
    if t and t.get('age_s', 1e9) <= max_age:
        age = t['age_s']
        when = 'just now' if age < 10 else f'{age:.0f} s ago' if age < 120 else \
            f'{age / 60:.0f} min ago'
        return (t['x'], t['y'], base[2]), f'last seen {when}'
    return base, 'not seen recently - trying its usual spot'


class Follower:
    """When to re-approach a followed animal, and when to give up. ROS-free (unit-tested).

    update() -> 'regoal' (walk to a new spot near it) | 'lost' | 'timeout' | None
    """

    def __init__(self, move=1.0, lost_after=30.0, max_time=300.0, min_interval=3.0,
                 look_after=4.0):
        self.move, self.lost_after, self.max_time = move, lost_after, max_time
        self.min_interval, self.look_after = min_interval, look_after
        self.name = None

    def start(self, name, xy, now):
        self.name, self.anchor, self.started, self.last_goal = name, xy, now, now
        self.looked = False

    def stop(self):
        self.name = None

    def update(self, track, goal_active, now):
        if self.name is None:
            return None
        if now - self.started > self.max_time:
            return 'timeout'
        if not track or track.get('age_s', 1e9) > self.lost_after:
            return 'lost'
        if track['age_s'] < 2.0:
            self.looked = False                 # in view again
        elif track['age_s'] > self.look_after and not self.looked and not goal_active:
            # out of the camera's 68 deg view (a deer walked round the side and the robot gave
            # up 2 m from it): turn a full circle to find it before giving up
            self.looked = True
            return 'look'
        moved = math.hypot(track['x'] - self.anchor[0], track['y'] - self.anchor[1])
        if moved > self.move and not goal_active and now - self.last_goal >= self.min_interval:
            self.anchor, self.last_goal = (track['x'], track['y']), now
            return 'regoal'
        return None


def freshest(seen, cls):
    """(name, track) of the most recently seen track of a class ('person 3' ...), or None."""
    tracks = [(n, t) for n, t in seen.items() if t.get('cls', n) == cls]
    return min(tracks, key=lambda nt: nt[1].get('age_s', 1e9)) if tracks else None


class Searcher:
    """Find the intruder: last-seen spot first (if recent), then the lookout route; at each
    stop turn a full circle so the camera sweeps around. ROS-free (unit-tested).

    next_step() -> ('go', name, (x, y)) | ('look',) | ('done',)
    """

    ROUTE = ('centre', 'lake', 'meadow', 'forest', 'ridge', 'home')

    def __init__(self, places, last_seen=None):
        stops = [(n, places[n][:2]) for n in self.ROUTE if n in places]
        if last_seen:
            stops.insert(0, ('last-seen spot', last_seen))
        self.plan = []
        for name, xy in stops:
            self.plan += [('go', name, xy), ('look',)]

    def next_step(self):
        return self.plan.pop(0) if self.plan else ('done',)


def parse_animals(world_text):
    """{name: (x, y, radius)} for every animal node in a .wbt file."""
    animals = {}
    kinds = '|'.join(k.capitalize() for k in ANIMAL_RADIUS)
    for kind, block in re.findall(rf'^({kinds})\s*\{{(.*?)^\}}', world_text, flags=re.S | re.M):
        t = re.search(r'translation\s+(\S+)\s+(\S+)', block)
        name = re.search(r'name\s+"([^"]*)"', block)
        if t:
            animals[(name.group(1) if name else kind).lower()] = (
                float(t.group(1)), float(t.group(2)), ANIMAL_RADIUS[kind.lower()])
    return animals


def parse_task(text, targets):
    """Plain words -> ('goto', target) | ('find', 'intruder') | ('patrol', None) |
    ('stop', None) | (None, reason)."""
    words = re.findall(r'[a-z]+', text.lower())
    if not words:
        return None, 'empty task'
    if 'stop' in words or 'halt' in words:
        return 'stop', None
    if 'patrol' in words:
        return 'patrol', None
    if set(words) & {'intruder', 'person', 'stranger', 'poacher'}:
        return 'find', 'intruder'       # "find / search for / follow the intruder"
    named = [ALIASES.get(w, w) for w in words if ALIASES.get(w, w) in targets]
    if not named:
        return None, f'no known place or animal in "{text}" (known: {", ".join(sorted(targets))})'
    return 'goto', named[-1]          # "go from the lake to the deer" -> deer


def approach_candidates(target, robot, standoff=1.0, steps=12):
    """Standing spots around target=(x, y, radius), nearest to the robot first.

    Each is (x, y, yaw) with yaw facing the target. A point target (radius 0) is its own goal.
    """
    tx, ty, radius = target
    if radius <= 0.0:
        yaw = math.atan2(ty - robot[1], tx - robot[0])
        return [(tx, ty, yaw)]
    d = radius + standoff
    base = math.atan2(robot[1] - ty, robot[0] - tx)       # direction target -> robot
    spots = []
    for k in range(steps):
        a = base + (k + 1) // 2 * (2 * math.pi / steps) * (1 if k % 2 else -1)
        x, y = tx + d * math.cos(a), ty + d * math.sin(a)
        spots.append((x, y, math.atan2(ty - y, tx - x)))
    return spots


class Tasks(Node):

    def __init__(self):
        super().__init__('tasks')
        world = os.path.expanduser(self.declare_parameter(
            'world', '~/ranger_ws/src/ranger_bot/worlds/nature_reserve.wbt').value)
        self.standoff = self.declare_parameter('standoff', 1.0).value
        self.targets = dict(PLACES)
        try:
            with open(world) as f:
                self.targets.update(parse_animals(f.read()))
        except OSError as e:
            self.get_logger().warn(f'No animals loaded ({e})')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.costmap = None
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.mode_pub = self.create_publisher(String, 'ranger/set_mode', 10)
        self.patrol_pub = self.create_publisher(String, 'ranger/patrol', 10)
        self.status_pub = self.create_publisher(String, 'ranger/task_status', 10)
        self.create_subscription(String, 'ranger/task', self.on_task, 10)
        self.create_subscription(
            OccupancyGrid, 'global_costmap/costmap', lambda m: setattr(self, 'costmap', m),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.pending = None           # (name, spot, send_at): goal sent once patrol has paused
        self.seen = {}                # /ranger/animals from animal_tracker.py
        self.seen_max_age = self.declare_parameter('seen_max_age', 600.0).value
        self.create_subscription(
            String, 'ranger/animals', self.on_animals,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.goal_handle = None
        self.goal_active = False
        self.follower = Follower(
            move=self.declare_parameter('follow_move', 1.0).value,
            lost_after=self.declare_parameter('follow_lost', 30.0).value,
            max_time=self.declare_parameter('follow_time', 300.0).value)
        self.person_standoff = self.declare_parameter('person_standoff', 2.5).value
        self.search = None            # Searcher while looking for the intruder
        self.spin = ActionClient(self, Spin, 'spin')
        self.spinning = False
        self.mode = None
        self.create_subscription(
            String, 'ranger/mode', self.on_mode,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'Tasks ready. Targets: {", ".join(sorted(self.targets))}')

    def status(self, text, warn=False):
        self.status_pub.publish(String(data=text))
        (self.get_logger().warn if warn else self.get_logger().info)(text)

    def on_mode(self, msg):
        prev, self.mode = self.mode, msg.data
        if prev == 'auto' and msg.data != 'auto' and (self.follower.name or self.search):
            self.get_logger().info('Follow/search ended (left AUTO)')
            self.follower.stop()
            self.search = None

    def on_task(self, msg):
        self.follower.stop()                    # any new command ends a follow / search
        self.search = None
        action, arg = parse_task(msg.data, self.targets)
        if action is None:
            self.status(f'Not understood: {arg}', warn=True)
        elif action == 'stop':
            self.mode_pub.publish(String(data='stop'))      # mode_manager halts in any mode
            self.status('Stopped - press L1/R1 again (or pick a mode) to drive')
        elif action == 'patrol':
            self.patrol_pub.publish(String(data='resume'))
            self.mode_pub.publish(String(data='auto'))
            self.status('Patrolling')
        elif action == 'find':
            self.start_search()
        else:
            self.go(arg)

    # ------------------------------------------------ find the intruder
    def track_of(self, name):
        """Latest track of a followed target ('intruder' = freshest person track)."""
        if name == 'intruder':
            f = freshest(self.seen, 'person')
            return f[1] if f else None
        return self.seen.get(name)

    def start_search(self):
        f = freshest(self.seen, 'person')
        last = (f[1]['x'], f[1]['y']) if f and f[1].get('age_s', 1e9) <= 120 else None
        self.search = Searcher(PLACES, last)
        self.patrol_pub.publish(String(data='pause'))
        self.mode_pub.publish(String(data='auto'))
        self.status('Searching for the intruder' + (' - going where it was last seen'
                                                    if last else ''))

    def search_tick(self):
        f = freshest(self.seen, 'person')
        if f and f[1].get('age_s', 1e9) < 3.0:  # found (the intruder node raises the alarm)
            t = f[1]
            self.search = None
            if self.goal_handle is not None and self.goal_active:
                self.goal_handle.cancel_goal_async()
            near = min(PLACES, key=lambda n: math.hypot(PLACES[n][0] - t['x'],
                                                        PLACES[n][1] - t['y']))
            self.follower.start('intruder', (t['x'], t['y']), time.monotonic())
            self.status(f'Found the intruder near the {near}, following')
            self.re_approach('intruder')
            return
        if self.goal_active or self.pending or self.spinning:
            return
        step = self.search.next_step()
        if step[0] == 'done':
            self.search = None
            self.status('Searched the reserve, no intruder found')
        elif step[0] == 'look':
            self.look_around()
        else:
            _, name, (x, y) = step
            robot = self.robot_xy()
            yaw = math.atan2(y - robot[1], x - robot[0]) if robot else 0.0
            self.get_logger().info(f'Searching: going to the {name}')
            self.send(name, (x, y, yaw), quiet=True)

    def look_around(self):
        """Turn a full circle on the spot (Nav2 Spin) so the camera sweeps around."""
        if self.spinning or not self.spin.server_is_ready():
            return
        goal = Spin.Goal()
        goal.target_yaw = 2 * math.pi
        goal.time_allowance.sec = 60
        self.spinning = True

        def accepted(f):
            h = f.result()
            if not h.accepted:
                self.spinning = False
                return
            h.get_result_async().add_done_callback(lambda _: setattr(self, 'spinning', False))
        self.spin.send_goal_async(goal).add_done_callback(accepted)

    def on_animals(self, msg):
        try:
            self.seen = json.loads(msg.data)
        except ValueError:
            pass

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time(),
                                                timeout=Duration(seconds=0.2))
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def free(self, x, y, max_cost=50):
        """Is (x, y) drivable in the global costmap (and not unknown)? True if no costmap yet."""
        m = self.costmap
        if m is None:
            return True
        i = int((x - m.info.origin.position.x) / m.info.resolution)
        j = int((y - m.info.origin.position.y) / m.info.resolution)
        if not (0 <= i < m.info.width and 0 <= j < m.info.height):
            return False
        c = m.data[j * m.info.width + i]
        return 0 <= c <= max_cost

    def go(self, name):
        robot = self.robot_xy()
        if robot is None:
            self.status('Cannot plan: robot position unknown', warn=True)
            return
        target, self.note = target_position(name, self.targets, self.seen, self.seen_max_age)
        spots = approach_candidates(target, robot, self.standoff)
        spot = next((s for s in spots if self.free(s[0], s[1])), None)
        if spot is None:
            self.status(f'No free spot to stand near the {name}', warn=True)
            return
        self.patrol_pub.publish(String(data='pause'))
        self.mode_pub.publish(String(data='auto'))
        self.pending = (name, spot, time.monotonic() + 0.7)   # let the patrol cancel first

    def tick(self):
        if self.pending and time.monotonic() >= self.pending[2]:
            name, spot, _ = self.pending
            self.pending = None
            self.send(name, spot)
        if self.search:
            self.search_tick()
        name = self.follower.name
        if name:
            act = self.follower.update(self.track_of(name),
                                       self.goal_active or self.pending or self.spinning,
                                       time.monotonic())
            if act == 'lost':
                self.follower.stop()
                self.status(f'Lost sight of the {name}')
            elif act == 'timeout':
                self.follower.stop()
                self.status(f'Stopped following the {name}')
            elif act == 'regoal':
                self.re_approach(name)
            elif act == 'look':
                self.get_logger().info(f'{name} out of view - looking around')
                self.look_around()

    def re_approach(self, name):
        """Quietly walk to a new standing spot next to a followed animal."""
        robot = self.robot_xy()
        if robot is None:
            return
        t = self.track_of(name)
        if t is None:
            return
        if name == 'intruder':                  # keep a respectful distance from people
            target, standoff = (t['x'], t['y'], 0.25), self.person_standoff
        else:
            target, standoff = (t['x'], t['y'], self.targets[name][2]), self.standoff
        spots = approach_candidates(target, robot, standoff)
        spot = next((s for s in spots if self.free(s[0], s[1])), None)
        if spot:
            self.get_logger().info(f'Following the {name}: now at ({t["x"]:.1f}, {t["y"]:.1f})')
            self.send(name, spot, quiet=True)

    def send(self, name, spot, quiet=False):
        if not self.nav.server_is_ready():
            self.status('Nav2 is not running', warn=True)
            return
        x, y, yaw = spot
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        if not quiet:
            note = f', {self.note}' if getattr(self, 'note', '') else ''
            self.status(f'Going to the {name} (standing at {x:.1f}, {y:.1f}{note})')
        self.goal_active = True
        self.nav.send_goal_async(goal).add_done_callback(
            lambda f: self.on_accepted(f, name, quiet))

    def on_accepted(self, future, name, quiet=False):
        handle = future.result()
        if not handle.accepted:
            self.goal_active = False
            if not quiet:
                self.status(f'Nav2 rejected the goal to the {name}', warn=True)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(f, name, quiet))

    def on_result(self, future, name, quiet=False):
        self.goal_active = False
        status = future.result().status
        # action_msgs/GoalStatus: 4 succeeded, 5 canceled, 6 aborted
        if quiet:                               # a follow step: keep following, retry later
            if status == 6:
                self.get_logger().info(f'Follow step to the {name} failed, will retry')
            return
        if status == 4 and name in ANIMAL_RADIUS:
            t = self.seen.get(name)
            xy = (t['x'], t['y']) if t else self.targets[name][:2]
            self.follower.start(name, xy, time.monotonic())
            self.status(f'Arrived at the {name}, following it')
            return
        text = {4: f'Arrived at the {name}', 5: f'Task to the {name} cancelled',
                6: f'Could not reach the {name}'}.get(status, f'Task to the {name} ended ({status})')
        self.status(text, warn=status != 4)


def main(args=None):
    rclpy.init(args=args)
    node = Tasks()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def send_main(args=None):
    """`ros2 run ranger_bot task "go towards the deer"`: publish one task, print the replies."""
    text = ' '.join(a for a in sys.argv[1:] if not a.startswith('--'))
    if not text:
        print('usage: ros2 run ranger_bot task "go towards the deer"')
        return
    rclpy.init(args=args)
    node = rclpy.create_node('task_cli')
    replies = []
    node.create_subscription(String, 'ranger/task_status', lambda m: replies.append(m.data), 10)
    pub = node.create_publisher(String, 'ranger/task', 10)
    end = time.time() + 5.0                     # wait for the tasks node to be discovered
    while pub.get_subscription_count() == 0 and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if pub.get_subscription_count() == 0:
        print('The tasks node is not running (start navigation.launch.py).')
    else:
        pub.publish(String(data=text))
        end = time.time() + 3.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        for r in replies:
            print(r)
        print('(follow progress: ros2 topic echo /ranger/task_status)')
    node.destroy_node()
    rclpy.shutdown()
