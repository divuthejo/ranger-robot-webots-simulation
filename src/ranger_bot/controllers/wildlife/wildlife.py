"""Webots supervisor: makes the reserve's animals wander and sends in an occasional intruder.

Runs inside Webots (the "wildlife" Robot node in nature_reserve.wbt), not as a ROS node.
Logic lives in ranger_bot/wildlife_sim.py. Ground truth for testing the robot's own estimates
is written to ~/.ros/wildlife_truth.json twice a second.

controllerArgs:  --seed N            (default: random)
                 --intruder-every S  seconds between intruder visits (default 300, 0 = never)
                 --intruder-first S  first visit after S seconds (default 120)
                 --still             animals stand still (the old static reserve)
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHARE = os.path.dirname(os.path.dirname(HERE))          # .../share/ranger_bot (maps/ next door)

from controller import Supervisor   # noqa: E402  (Webots)

from ranger_bot.tasks import ANIMAL_RADIUS   # noqa: E402
from ranger_bot.wildlife_sim import (Intruder, Walker, WalkGrid,   # noqa: E402
                                     entry_points, rng_from)


def arg(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


def main():
    sup = Supervisor()
    dt_ms = int(sup.getBasicTimeStep())
    seed = arg('--seed', None)
    rng = rng_from(int(seed) if seed is not None else None)
    every = float(arg('--intruder-every', 300))
    first = float(arg('--intruder-first', 120))
    still = '--still' in sys.argv

    children = sup.getRoot().getField('children')
    animals, robot_node, person_node = [], None, None
    for k in range(children.getCount()):
        n = children.getMFNode(k)
        kind = n.getTypeName().lower()
        if kind in ANIMAL_RADIUS:
            animals.append((n, kind))
        elif kind == 'turtlebot3burger':
            robot_node = n
        elif kind == 'pedestrian':
            person_node = n

    walkers = []
    for n, kind in animals:
        x, y, _ = n.getField('translation').getSFVec3f()
        ax, ay, az, ang = n.getField('rotation').getSFRotation()
        yaw = ang if az >= 0 else -ang
        name = n.getField('name').getSFString()
        walkers.append((n, Walker(name, kind, x, y, yaw, ANIMAL_RADIUS[kind], rng)))

    maps = os.path.join(SHARE, 'maps')
    t0 = time.time()
    grid = WalkGrid.from_files(os.path.join(maps, 'reserve.yaml'),
                               os.path.join(maps, 'keepout.yaml'),
                               clear=[(w.x, w.y, w.radius + 0.25) for _, w in walkers])
    print(f'[wildlife] {len(walkers)} animals, walk grid ready in {time.time() - t0:.1f} s'
          f'{" (still)" if still else ""}; intruder every {every:.0f} s'
          f'{"" if person_node else " - no Pedestrian in the world, intruder disabled"}',
          flush=True)

    entries = entry_points(grid)
    intruder, next_visit = None, first if every > 0 else float('inf')
    stuck_ref = None
    person_fields = {}
    if person_node:
        hide = person_node.getField('translation')
        hide.setSFVec3f([0.0, 0.0, -10.0])               # parked under the ground until needed
        for f in ('leftLegAngle', 'rightLegAngle', 'leftArmAngle', 'rightArmAngle',
                  'leftLowerLegAngle', 'rightLowerLegAngle'):
            fld = person_node.getField(f)
            if fld is not None:
                person_fields[f] = fld
        if not person_fields:
            print('[wildlife] Pedestrian limb fields not accessible - it will glide', flush=True)

    truth_path = os.path.expanduser('~/.ros/wildlife_truth.json')
    os.makedirs(os.path.dirname(truth_path), exist_ok=True)
    last_write, logic_dt, acc = -1.0, 0.1, 0.0

    while sup.step(dt_ms) != -1:
        t = sup.getTime()
        acc += dt_ms / 1000.0
        if acc < logic_dt:
            continue
        step_dt, acc = acc, 0.0
        robot = None
        if robot_node:
            rx, ry, _ = robot_node.getPosition()
            robot = (rx, ry)

        if not still:
            ws = [w for _, w in walkers]
            for n, w in walkers:
                w.step(t, step_dt, grid, robot,
                       [o for o in ws if o is not w] + ([intruder] if intruder else []))
                n.getField('translation').setSFVec3f([w.x, w.y, 0.0])
                n.getField('rotation').setSFRotation([0.0, 0.0, 1.0, w.yaw])

        if person_node:
            if intruder is None and t >= next_visit and entries:
                intruder = Intruder(rng.choice(entries), rng, exits=entries)
                print(f'[wildlife] intruder enters at ({intruder.x:.1f}, {intruder.y:.1f})',
                      flush=True)
            if intruder is not None:
                intruder.step(t, step_dt, grid, robot, [])
                # watchdog: a frozen intruder kept the robot on watch for 6+ min (cause unknown
                # yet - its state is in the truth file and logged here)
                if stuck_ref is None or math.hypot(intruder.x - stuck_ref[0],
                                                   intruder.y - stuck_ref[1]) > 0.3:
                    stuck_ref = (intruder.x, intruder.y, t)
                elif t - stuck_ref[2] > 90.0:
                    print(f'[wildlife] intruder stuck 90 s at ({intruder.x:.2f}, {intruder.y:.2f})'
                          f' state={intruder.state} goal={intruder.goal} left={intruder.left}'
                          f' until={intruder.until:.1f} t={t:.1f} - removing it', flush=True)
                    intruder.gone = True
                person_node.getField('translation').setSFVec3f([intruder.x, intruder.y, 1.27])
                person_node.getField('rotation').setSFRotation([0.0, 0.0, 1.0, intruder.yaw])
                for f, v in intruder.phase_angles(t).items():
                    if f in person_fields:
                        person_fields[f].setSFFloat(v)
                if intruder.gone:
                    print('[wildlife] intruder left the reserve', flush=True)
                    person_node.getField('translation').setSFVec3f([0.0, 0.0, -10.0])
                    intruder, next_visit, stuck_ref = None, t + every, None

        if t - last_write >= 0.5:
            last_write = t
            truth = {'time': round(t, 2),
                     'animals': {w.name: [round(w.x, 3), round(w.y, 3), round(w.yaw, 3)]
                                 for _, w in walkers},
                     'intruder': [round(intruder.x, 3), round(intruder.y, 3)]
                     if intruder else None,
                     'intruder_state': {'state': intruder.state, 'goal': intruder.goal,
                                        'left': intruder.left, 'until': round(intruder.until, 1),
                                        'speed': intruder.speed} if intruder else None}
            tmp = truth_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(truth, f)
            os.replace(tmp, truth_path)


main()
