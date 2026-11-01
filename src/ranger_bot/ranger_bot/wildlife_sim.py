"""Wandering animals and an occasional intruder for the Webots reserve - the pure logic.

The Webots supervisor controller (controllers/wildlife/wildlife.py) only reads poses and writes
them back each step; everything here is plain Python so it can be unit-tested.

Animals are kinematic (Webots Solids without Physics): moving one INTO the robot makes ODE
fling the robot (the old "physics step could not be computed" blow-ups), so the one hard rule
is that no step ever ends closer to the robot than ROBOT_CLEAR. Other rules:
  - each animal keeps to a home range around its start spot (the zones stay recognisable),
  - it grazes (stands still) for a while, then walks to a random free spot, then grazes again,
  - it only walks on free map cells, clear of trees/walls/the lake and of the low rocks in the
    keepout mask, with its own radius as clearance,
  - it keeps apart from other animals, and walks away when the robot comes close (wildlife!),
    but not so early that the robot can never stand 1 m from it (tasks.py standoff).
The intruder (a Pedestrian) walks in from the edge of the reserve, wanders and leaves again.
"""

import math
import random
from collections import deque

ROBOT_RADIUS = 0.12             # TurtleBot3 Burger footprint
ROBOT_CLEAR = 0.35              # min gap between robot footprint and an animal's edge
ANIMAL_GAP = 0.8                # min gap between two animals' edges

# speed m/s, home range m, flee when the robot is within this of the animal's edge, graze s
SPECIES = {
    'deer':   dict(speed=0.25, home=4.0, flee=0.6, graze=(6, 20)),
    'horse':  dict(speed=0.30, home=4.0, flee=0.5, graze=(8, 25)),
    'cow':    dict(speed=0.15, home=3.0, flee=0.4, graze=(10, 30)),
    'sheep':  dict(speed=0.15, home=3.0, flee=0.6, graze=(8, 25)),
    'fox':    dict(speed=0.30, home=5.0, flee=0.8, graze=(4, 12)),
    'rabbit': dict(speed=0.35, home=3.0, flee=0.8, graze=(3, 10)),
    'dog':    dict(speed=0.30, home=5.0, flee=0.5, graze=(4, 15)),
    'cat':    dict(speed=0.20, home=3.0, flee=0.7, graze=(5, 15)),
}
INTRUDER = dict(speed=0.5, radius=0.3, wander=(3, 6))   # waypoints inside before leaving
TURN_RATE = 1.5                 # rad/s


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def read_pgm(path):
    """(width, height, bytes) of a binary P5 PGM (row 0 = top of the map)."""
    with open(path, 'rb') as f:
        data = f.read()
    fields, pos = [], 0
    while len(fields) < 4:                      # magic, width, height, maxval (+ comments)
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos)
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    w, h = int(fields[1]), int(fields[2])
    return w, h, data[pos + 1:pos + 1 + w * h]


def read_map_yaml(path):
    """(image file, resolution, origin x, origin y) - just the fields we need, no yaml lib."""
    vals = {}
    for line in open(path):
        if ':' in line and not line.lstrip().startswith('#'):
            k, v = line.split(':', 1)
            vals[k.strip()] = v.split('#')[0].strip()
    ox, oy = [float(v) for v in vals['origin'].strip('[]').split(',')[:2]]
    return vals['image'], float(vals['resolution']), ox, oy


class WalkGrid:
    """Clearance map: distance (m) from each cell to the nearest blocked cell.

    blocked = anything not free in the SLAM map (walls, trees, the unmapped lake) plus rocks in
    the keepout mask. `clear` circles (x, y, r) are forced free first: the animals were standing
    there while mapping, so their starting spots are 'occupied' in the SLAM map.
    """

    def __init__(self, width, height, resolution, ox, oy, blocked, clear=()):
        self.w, self.h, self.res, self.ox, self.oy = width, height, resolution, ox, oy
        blocked = bytearray(blocked)
        for cx, cy, r in clear:
            ci, cj = self.cell(cx, cy)
            n = int(r / resolution) + 1
            for j in range(max(0, cj - n), min(height, cj + n + 1)):
                for i in range(max(0, ci - n), min(width, ci + n + 1)):
                    if math.hypot(i - ci, j - cj) * resolution <= r:
                        blocked[j * width + i] = 0
        self.dist = self._brushfire(blocked)

    @classmethod
    def from_files(cls, map_yaml, keepout_yaml=None, clear=()):
        import os
        img, res, ox, oy = read_map_yaml(map_yaml)
        w, h, px = read_pgm(os.path.join(os.path.dirname(map_yaml), img))
        blocked = bytearray(0 if v >= 250 else 1 for v in px)       # 254 free, 205/0 blocked
        if keepout_yaml and os.path.exists(keepout_yaml):
            kimg, _, _, _ = read_map_yaml(keepout_yaml)
            kw, kh, kpx = read_pgm(os.path.join(os.path.dirname(keepout_yaml), kimg))
            if (kw, kh) == (w, h):
                for k, v in enumerate(kpx):
                    if v < 50:
                        blocked[k] = 1
        # pgm row 0 is the top (largest y): flip so row j is y = oy + j*res
        flipped = bytearray(w * h)
        for j in range(h):
            flipped[j * w:(j + 1) * w] = blocked[(h - 1 - j) * w:(h - j) * w]
        return cls(w, h, res, ox, oy, flipped, clear)

    def _brushfire(self, blocked):
        w, h = self.w, self.h
        inf = 1e9
        dist = [inf] * (w * h)
        q = deque()
        for k, b in enumerate(blocked):
            if b:
                dist[k] = 0.0
                q.append(k)
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while q:                                # chamfer distance, good to a few cm
            k = q.popleft()
            j, i = divmod(k, w)
            d = dist[k]
            for di, dj, c in steps:
                ii, jj = i + di, j + dj
                if 0 <= ii < w and 0 <= jj < h:
                    kk = jj * w + ii
                    if dist[kk] > d + c:
                        dist[kk] = d + c
                        q.append(kk)
        return [d * self.res for d in dist]

    def cell(self, x, y):
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def clearance(self, x, y):
        i, j = self.cell(x, y)
        if not (0 <= i < self.w and 0 <= j < self.h):
            return 0.0
        return self.dist[j * self.w + i]

    def free(self, x, y, r):
        return self.clearance(x, y) >= r

    def segment_free(self, x0, y0, x1, y1, r, step=0.1):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        return all(self.free(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n, r)
                   for t in range(n + 1))


class Walker:
    """One animal (or the intruder): pose + graze/walk/flee state machine."""

    def __init__(self, name, kind, x, y, yaw, radius, rng, params=None):
        self.name, self.kind, self.radius, self.rng = name, kind, radius, rng
        self.x, self.y, self.yaw = x, y, yaw
        self.home = (x, y)
        self.p = params or SPECIES[kind]
        self.state, self.until, self.goal = 'graze', rng.uniform(*self.p['graze']), None
        self.speed = 0.0

    # ---------------------------------------------------------- helpers
    def gap_to(self, x, y, r, px=None, py=None):
        px, py = (self.x, self.y) if px is None else (px, py)
        return math.hypot(px - x, py - y) - self.radius - r

    def ok_at(self, x, y, grid, robot, others):
        """May this animal stand at (x, y)? A spot too close to the robot or another animal is
        still allowed if it is FARTHER than where the animal is now: when the robot has already
        driven in close (its lidar only sees a horse's legs, so Nav2 happily drives between
        them), the animal must be able to walk away - freezing it let the robot get under it."""
        if not grid.free(x, y, self.radius):
            return False
        if robot:
            new = self.gap_to(robot[0], robot[1], ROBOT_RADIUS, x, y)
            if new < ROBOT_CLEAR and new <= self.gap_to(robot[0], robot[1], ROBOT_RADIUS) + 1e-9:
                return False
        for o in others:
            new = self.gap_to(o.x, o.y, o.radius, x, y)
            if new < ANIMAL_GAP and new <= self.gap_to(o.x, o.y, o.radius) + 1e-9:
                return False
        return True

    def pick_goal(self, grid, robot, others, center=None, spread=None, tries=40):
        g = self._pick(grid, robot, others, center or self.home, spread or self.p['home'], tries)
        if g is None:
            # in a tight pocket (forest) almost every long straight line hits a tree, so random
            # far goals kept failing and the intruder stood still for 6+ min: hop out locally
            g = self._pick(grid, robot, others, (self.x, self.y), 2.0, tries)
        return g

    def _pick(self, grid, robot, others, center, spread, tries):
        cx, cy = center
        for _ in range(tries):
            a, d = self.rng.uniform(-math.pi, math.pi), spread * math.sqrt(self.rng.random())
            gx, gy = cx + d * math.cos(a), cy + d * math.sin(a)
            if math.hypot(gx - self.x, gy - self.y) < 0.5:
                continue
            if self.ok_at(gx, gy, grid, robot, others) and \
                    grid.segment_free(self.x, self.y, gx, gy, self.radius):
                return gx, gy
        return None

    def flee_goal(self, grid, robot, others):
        away = math.atan2(self.y - robot[1], self.x - robot[0])
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):     # straight away first
            gx = self.x + 1.5 * math.cos(away + off)
            gy = self.y + 1.5 * math.sin(away + off)
            if self.ok_at(gx, gy, grid, robot, others) and \
                    grid.segment_free(self.x, self.y, gx, gy, self.radius):
                return gx, gy
        return None

    # ---------------------------------------------------------- step
    def step(self, t, dt, grid, robot, others):
        """Advance by dt. robot = (x, y) or None. others = the other Walkers."""
        if robot and self.state != 'flee' and 'flee' in self.p and \
                self.gap_to(robot[0], robot[1], ROBOT_RADIUS) < self.p['flee']:
            g = self.flee_goal(grid, robot, others)
            if g:
                self.state, self.goal = 'flee', g
        if self.state == 'graze':
            self.speed = 0.0
            if t >= self.until:
                g = self.pick_goal(grid, robot, others)
                if g:
                    self.state, self.goal = 'walk', g
                else:
                    self.until = t + 2.0
            return
        gx, gy = self.goal
        dist = math.hypot(gx - self.x, gy - self.y)
        if dist < 0.1:
            self.rest(t)
            return
        want = math.atan2(gy - self.y, gx - self.x)
        err = wrap(want - self.yaw)
        self.yaw = wrap(self.yaw + max(-TURN_RATE * dt, min(TURN_RATE * dt, err)))
        if abs(err) > 0.4:                      # turn on the spot first
            self.speed = 0.0
            return
        v = self.p['speed'] * (1.5 if self.state == 'flee' else 1.0)
        d = min(v * dt, dist)
        nx, ny = self.x + d * math.cos(self.yaw), self.y + d * math.sin(self.yaw)
        if not self.ok_at(nx, ny, grid, robot, others):
            self.speed = 0.0                    # blocked (robot / animal in the way): wait
            self.rest(t, short=True)
            return
        self.x, self.y, self.speed = nx, ny, v

    def rest(self, t, short=False):
        self.state, self.goal = 'graze', None
        lo, hi = (1.5, 4.0) if short else self.p['graze']
        self.until = t + self.rng.uniform(lo, hi)


class Intruder(Walker):
    """A person: walks in from an entry point on the edge, visits a few spots, walks out."""

    def __init__(self, entry, rng, exits=()):
        super().__init__('intruder', 'person', entry[0], entry[1], 0.0, INTRUDER['radius'], rng,
                         params=dict(speed=INTRUDER['speed'], home=6.0, graze=(2, 6)))
        self.entry = entry
        self.exits = list(exits) or [entry]
        self.left = rng.randint(*INTRUDER['wander'])
        self.gone = False

    def step(self, t, dt, grid, robot, others):
        if self.gone:
            return
        if self.state == 'graze' and t >= self.until:
            if self.left <= 0:                  # head back out, along a clear straight line:
                self.left = -1                  # a blocked line left it stuck by a tree forever
                if not grid.segment_free(self.x, self.y, *self.entry, self.radius):
                    exits = [e for e in self.exits if
                             grid.segment_free(self.x, self.y, e[0], e[1], self.radius)]
                    if exits:
                        self.entry = min(exits, key=lambda e: math.hypot(e[0] - self.x,
                                                                         e[1] - self.y))
                    else:                       # no exit in sight: wander a bit, try again
                        g = self.pick_goal(grid, robot, [], center=(self.x, self.y), spread=4.0)
                        if g:
                            self.state, self.goal = 'walk', g
                        else:
                            self.until = t + 2.0
                        return
                self.state, self.goal = 'walk', self.entry
            else:
                g = self.pick_goal(grid, robot, [], center=(0.0, 0.0), spread=11.0)
                if g:
                    self.state, self.goal = 'walk', g
                    self.left -= 1
                else:
                    self.until = t + 2.0
            return
        was_leaving = self.left < 0
        super().step(t, dt, grid, robot, [])
        if was_leaving and self.state == 'walk' and self.goal != self.entry:
            pass                                # a detour on the way out
        elif was_leaving and math.hypot(self.x - self.entry[0], self.y - self.entry[1]) < 0.15:
            self.gone = True                    # reached the exit (not merely blocked)

    def phase_angles(self, t):
        """Leg/arm swing for the Pedestrian hidden fields while walking (rad)."""
        if self.speed <= 0.0:
            return dict.fromkeys(('leftLegAngle', 'rightLegAngle', 'leftArmAngle',
                                  'rightArmAngle', 'leftLowerLegAngle',
                                  'rightLowerLegAngle'), 0.0)
        s = math.sin(2 * math.pi * 1.6 * t)
        return {'leftLegAngle': 0.45 * s, 'rightLegAngle': -0.45 * s,
                'leftArmAngle': -0.35 * s, 'rightArmAngle': 0.35 * s,
                'leftLowerLegAngle': max(0.0, -0.6 * s), 'rightLowerLegAngle': max(0.0, 0.6 * s)}


def entry_points(grid, r=INTRUDER['radius'], margin=1.0, half=15.0):
    """Free spots just inside the reserve's fence, all round, for the intruder to enter at."""
    pts = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        for inset in (margin, margin + 0.5, margin + 1.0):
            x = max(-half + inset, min(half - inset, 20 * math.cos(a)))
            y = max(-half + inset, min(half - inset, 20 * math.sin(a)))
            if grid.free(x, y, r):
                pts.append((x, y))
                break
    return pts


def rng_from(seed):
    return random.Random(seed)
