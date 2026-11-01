"""wildlife_sim: animals wander safely on the real reserve map (no Webots)."""

import math
import os

import pytest

from ranger_bot.tasks import ANIMAL_RADIUS, parse_animals
from ranger_bot.wildlife_sim import (ROBOT_CLEAR, ROBOT_RADIUS, Intruder, Walker, WalkGrid,
                                     entry_points, read_pgm, rng_from)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS = os.path.join(PKG, 'maps')


@pytest.fixture(scope='module')
def world():
    with open(os.path.join(PKG, 'worlds', 'nature_reserve.wbt')) as f:
        animals = parse_animals(f.read())
    grid = WalkGrid.from_files(os.path.join(MAPS, 'reserve.yaml'),
                               os.path.join(MAPS, 'keepout.yaml'),
                               clear=[(x, y, r + 0.25) for x, y, r in animals.values()])
    return grid, animals


def make_walkers(animals, seed=1):
    rng = rng_from(seed)
    kinds = {n: n for n in animals}             # in this world name == kind
    return [Walker(n, kinds[n], x, y, 0.0, ANIMAL_RADIUS[kinds[n]], rng)
            for n, (x, y, _) in animals.items()]


def run(walkers, grid, robot_path, seconds, dt=0.1, check=None, intruder=None):
    t = 0.0
    while t < seconds:
        robot = robot_path(t)
        for w in walkers:
            w.step(t, dt, grid, robot, [o for o in walkers if o is not w] +
                   ([intruder] if intruder else []))
        if intruder:
            intruder.step(t, dt, grid, robot, [])
        if check:
            check(t, robot)
        t += dt


def test_pgm_reader():
    w, h, px = read_pgm(os.path.join(MAPS, 'reserve.pgm'))
    assert (w, h) == (601, 600) and len(px) == w * h


def test_start_spots_are_walkable(world):
    grid, animals = world
    for name, (x, y, r) in animals.items():
        assert grid.free(x, y, r), name
    assert not grid.free(8.0, 7.0, 0.1)          # the lake centre
    assert not grid.free(4.3, 8.6, 0.1)          # lake rock 1 (keepout mask)


def test_animals_move_stay_walkable_and_in_home_range(world):
    grid, animals = world
    ws = make_walkers(animals)

    def check(t, robot):
        for w in ws:
            assert grid.free(w.x, w.y, w.radius - 1e-6), (w.name, t, w.x, w.y)
            assert math.hypot(w.x - w.home[0], w.y - w.home[1]) <= w.p['home'] + 0.6, w.name
    run(ws, grid, lambda t: None, 600, check=check)
    moved = [w.name for w in ws if math.hypot(w.x - w.home[0], w.y - w.home[1]) > 0.5]
    assert len(moved) >= 6, moved


def test_animals_never_close_to_robot_driving_through_them(world):
    grid, animals = world
    ws = make_walkers(animals, seed=7)
    # robot visits each animal's start spot in turn at 0.2 m/s, i.e. drives right at them
    stops = [(0.0, 0.0)] + [(x, y) for x, y, _ in animals.values()]
    legs = []
    for a, b in zip(stops, stops[1:]):
        legs.append((a, b, math.hypot(b[0] - a[0], b[1] - a[1]) / 0.2))

    def robot_path(t):
        for (ax, ay), (bx, by), dur in legs:
            if t < dur:
                f = t / dur
                return ax + (bx - ax) * f, ay + (by - ay) * f
            t -= dur
        return stops[-1]
    total = sum(d for _, _, d in legs)
    t, dt, bad, moves = 0.0, 0.1, [], 0
    while t < total:
        robot = robot_path(t)
        for w in ws:
            before = math.hypot(w.x - robot[0], w.y - robot[1])
            w.step(t, dt, grid, robot, [o for o in ws if o is not w])
            after = math.hypot(w.x - robot[0], w.y - robot[1])
            if after < before - 1e-9:
                moves += 1
                if after - w.radius - ROBOT_RADIUS < ROBOT_CLEAR - 1e-9:
                    bad.append((w.name, round(t, 1), round(after - w.radius - ROBOT_RADIUS, 2)))
        t += dt
    # The robot may drive INTO a standing animal (Nav2's job to avoid), but an animal's own
    # step never brings it closer to the robot than ROBOT_CLEAR.
    assert moves > 100 and not bad, bad[:5]


def test_a_step_never_ends_closer_than_robot_clear(world):
    grid, animals = world
    rng = rng_from(3)
    x, y, r = animals['deer']
    w = Walker('deer', 'deer', x, y, 0.0, r, rng)
    robot = (x + r + ROBOT_RADIUS + 1.0, y)      # 1 m east of the deer's edge
    for k in range(3000):
        before = math.hypot(w.x - robot[0], w.y - robot[1])
        w.step(k * 0.1, 0.1, grid, robot, [])
        after = math.hypot(w.x - robot[0], w.y - robot[1])
        if after < before:                        # moved towards the robot: must stay clear
            assert after - r - ROBOT_RADIUS >= ROBOT_CLEAR - 1e-9


def test_flees_when_robot_comes_close(world):
    grid, animals = world
    x, y, r = animals['rabbit']
    w = Walker('rabbit', 'rabbit', x, y, 0.0, r, rng_from(5))
    w.until = 1e9                                   # would graze forever
    robot = (x + r + ROBOT_RADIUS + 0.5, y)         # 0.5 m from its edge (< flee 0.8)
    for k in range(100):
        w.step(k * 0.1, 0.1, grid, robot, [])
    assert math.hypot(w.x - robot[0], w.y - robot[1]) > 0.5 + r + ROBOT_RADIUS + 0.5


def test_large_animal_lets_robot_stand_at_task_standoff(world):
    """tasks.py stops 1.0 m from the edge: the deer must not run from that."""
    grid, animals = world
    x, y, r = animals['deer']
    w = Walker('deer', 'deer', x, y, 0.0, r, rng_from(2))
    w.until = 1e9
    robot = (x - r - ROBOT_RADIUS - 1.0, y)
    for k in range(50):
        w.step(k * 0.1, 0.1, grid, robot, [])
    assert (w.x, w.y) == (x, y)


def test_intruder_enters_wanders_and_leaves(world):
    grid, _ = world
    entries = entry_points(grid)
    assert len(entries) >= 20
    for x, y in entries:
        assert abs(x) <= 15 and abs(y) <= 15
    rng = rng_from(11)
    p = Intruder(rng.choice(entries), rng)
    far = max(math.hypot(p.x, p.y) for _ in [0])
    t = 0.0
    while not p.gone and t < 1500:
        p.step(t, 0.1, grid, None, [])
        far = min(far, math.hypot(p.x, p.y))
        t += 0.1
    assert p.gone, f'still inside after {t:.0f} s at ({p.x:.1f}, {p.y:.1f})'
    assert far < 12                                  # it really came into the reserve


def test_intruder_limbs_swing_only_while_walking(world):
    p = Intruder((0.0, 0.0), rng_from(1))
    assert all(v == 0.0 for v in p.phase_angles(1.0).values())
    p.speed = 0.5
    assert any(abs(v) > 0.1 for v in p.phase_angles(0.15).values())


def test_animal_escapes_when_robot_already_under_it(world):
    """Regression 2026-11-01: Nav2 drove between the horse's legs (the lidar sees only legs);
    the clearance rule then forbade every step, so the horse froze over the robot."""
    grid, animals = world
    x, y, r = animals['horse']
    w = Walker('horse', 'horse', x, y, 0.0, r, rng_from(4))
    w.until = 1e9
    robot = (x + 0.7, y)                            # inside the horse's 0.9 m circle
    start_gap = w.gap_to(robot[0], robot[1], ROBOT_RADIUS)
    for k in range(150):
        before = w.gap_to(robot[0], robot[1], ROBOT_RADIUS)
        w.step(k * 0.1, 0.1, grid, robot, [])
        assert w.gap_to(robot[0], robot[1], ROBOT_RADIUS) >= before - 1e-9   # never closer
    assert start_gap < 0 and w.gap_to(robot[0], robot[1], ROBOT_RADIUS) > ROBOT_CLEAR


def test_intruder_leaves_even_when_its_exit_line_is_blocked(world):
    """Regression: walking straight back to its entry it got stuck behind a tree for good."""
    grid, _ = world
    entries = entry_points(grid)
    rng = rng_from(21)
    p = Intruder(entries[0], rng, exits=entries)
    p.x, p.y, p.left = 1.68, 8.6, 0                # where it was stuck, done wandering
    p.entry = (14.0, 12.2)
    t = 0.0
    while not p.gone and t < 900:
        p.step(t, 0.1, grid, None, [])
        t += 0.1
    assert p.gone, (p.x, p.y)


def test_escapes_a_tight_forest_pocket(world):
    """Regression: from (-10.25, 1.13) (clearance 0.31 m) random far goals nearly all fail."""
    grid, _ = world
    moved = 0
    for seed in range(20):
        p = Intruder((14.0, 5.2), rng_from(seed), exits=entry_points(grid))
        p.x, p.y, p.left, p.until = -10.248, 1.132, 3, 0.0
        for k in range(300):                         # 30 s
            p.step(k * 0.1, 0.1, grid, None, [])
        moved += math.hypot(p.x + 10.248, p.y - 1.132) > 1.0
    assert moved == 20
