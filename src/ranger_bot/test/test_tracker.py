"""animal_tracker geometry + tracking, with synthetic boxes and lidar beams."""

import math

from ranger_bot.animal_tracker import (CAM_HEIGHT, CAM_X, CY, FX, LIDAR_X, RADIUS, Tracker,
                                       bearing, ground_range, locate)


def project(pose, ax, ay, radius, height=0.8):
    """Box (u1, v1, u2, v2) of an animal (cylinder) at map (ax, ay) seen from robot pose."""
    x, y, yaw = pose
    cx, cy = x + CAM_X * math.cos(yaw), y + CAM_X * math.sin(yaw)
    d = math.hypot(ax - cx, ay - cy)
    b = math.atan2(ay - cy, ax - cx) - yaw
    half = math.atan2(radius, d)
    u = lambda ang: 320 - FX * math.tan(ang)
    near = d - radius
    v_bottom = CY + CAM_HEIGHT * FX / near
    v_top = CY - (height - CAM_HEIGHT) * FX / near
    return (u(b + half), v_top, u(b - half), v_bottom)


def beams_for(pose, ax, ay, radius, n=360):
    """Lidar returns (base_link angle, range) from a circle at (ax, ay); nothing else."""
    x, y, yaw = pose
    lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
    out = []
    for i in range(n):
        a = -math.pi + 2 * math.pi * i / n
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        fx, fy = lx - ax, ly - ay                        # ray-circle intersection
        bq = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = bq * bq - 4 * c
        if disc >= 0:
            t = (-bq - math.sqrt(disc)) / 2
            if 0.12 < t < 3.5:
                out.append((a, t))
    return out


def test_bearing_and_ground_range():
    assert bearing(320) == 0 and bearing(0) > 0.59 and bearing(640) < -0.59
    assert abs(ground_range(CY + CAM_HEIGHT * FX / 2.0) - 2.0) < 1e-9
    assert ground_range(CY) is None and ground_range(479) is None


def test_locate_with_lidar_close_animal():
    pose = (1.0, 2.0, 0.7)
    ax, ay, r = 3.0, 3.5, RADIUS['deer']
    box = project(pose, ax, ay, r)
    x, y, rng, src = locate('deer', box, beams_for(pose, ax, ay, r), pose)
    assert src == 'lidar'
    assert math.hypot(x - ax, y - ay) < 0.3


def test_locate_small_animal_under_lidar_uses_ground():
    pose = (0.0, 0.0, -1.2)
    ax, ay = 1.5, -3.5
    r = RADIUS['rabbit']
    box = project(pose, ax, ay, r, height=0.15)
    x, y, rng, src = locate('rabbit', box, [], pose)       # lidar passes over it
    assert src == 'ground'
    assert math.hypot(x - ax, y - ay) < 0.35


def test_lidar_hitting_a_tree_behind_is_rejected():
    pose = (0.0, 0.0, 0.0)
    ax, ay, r = 2.5, 0.0, RADIUS['fox']
    box = project(pose, ax, ay, r, height=0.3)
    tree = beams_for(pose, 6.0, 0.0, 0.3)                  # lidar return far behind the fox
    x, y, rng, src = locate('fox', box, tree, pose)
    assert src == 'ground' and abs(x - ax) < 0.4


def test_far_animal_with_no_range_is_skipped():
    pose = (0.0, 0.0, 0.0)
    box = (300, 230, 310, 241)                             # bottom at the horizon
    assert locate('deer', box, [], pose) is None


def test_tracker_confirms_follows_and_names():
    tr = Tracker()
    for k in range(3):
        ev, t = tr.update('deer', 4.0 + 0.05 * k, 5.5, 0.9, 10.0 + k)
    assert ev == 'confirmed'
    for k in range(20):                                     # deer walks 2 m east
        tr.update('deer', 4.1 + 0.1 * k, 5.5, 0.9, 13.0 + k)
    snap = tr.snapshot(40.0)
    assert list(snap) == ['deer'] and abs(snap['deer']['x'] - 6.0) < 0.3
    assert snap['deer']['age_s'] == 8.0


def test_tracker_two_of_a_kind_and_flicker_ignored():
    tr = Tracker()
    for k in range(3):
        tr.update('sheep', 0.0, 0.0, 0.8, k)
        tr.update('sheep', 6.0, 0.0, 0.8, k)
    tr.update('fox', 3.0, 3.0, 0.6, 1.0)                    # one-frame flicker
    tr.prune(20.0)
    snap = tr.snapshot(20.0)
    assert set(snap) == {'sheep', 'sheep 2'}


def test_refound_after_a_minute():
    tr = Tracker()
    for k in range(3):
        tr.update('cat', 5.0, -9.5, 0.7, k)
    ev, _ = tr.update('cat', 5.5, -9.0, 0.7, 100.0)
    assert ev == 'refound'


def test_body_points_fill_fresh_tracks_and_keep_robot_clear():
    from ranger_bot.animal_tracker import body_points
    tracks = {'horse': {'cls': 'horse', 'x': 0.0, 'y': 0.0, 'age_s': 1.0},
              'cat': {'cls': 'cat', 'x': 5.0, 'y': 5.0, 'age_s': 60.0}}       # stale: no points
    pts = body_points(tracks)
    assert pts and all(math.hypot(x, y) <= RADIUS['horse'] + 1e-9 for x, y, _ in pts)
    assert not any(math.hypot(x - 5, y - 5) < 1 for x, y, _ in pts)
    near = body_points(tracks, robot=(0.6, 0.0))                              # robot under it
    assert near and all(math.hypot(x - 0.6, y) >= 0.35 for x, y, _ in near)


def test_lidar_off_a_tree_in_front_of_a_far_person_is_rejected():
    pose = (0.0, 0.0, 0.0)
    box = (302, 188, 320, 257)                      # person ~11 m away behind a trunk
    tree = beams_for(pose, 2.5, 0.0, 0.15)          # trunk 2.5 m ahead, inside the box
    x, y, rng, src = locate('person', box, tree, pose)
    assert src == 'height' and 9.5 < rng < 12.5


def test_person_range_from_known_height():
    pose = (0.0, 0.0, 0.0)
    d = 9.0
    box = (310, CY - (1.7 - CAM_HEIGHT) * FX / d, 330, CY + CAM_HEIGHT * FX / d)
    x, y, rng, src = locate('person', box, [], pose)
    assert src == 'height' and abs(x - d) < 0.4
