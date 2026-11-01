"""Collision-avoiding correction of a driver's velocity command (no ROS imports, so it is unit-testable).

The idea is the one used in smart wheelchairs: the user stays in charge, and the
robot only changes the command when following it would lead into an obstacle.

    correct(v, w, obstacles) -> (v', w', intervention)

Candidates (v', w') around the user's (v, w) are rolled out as constant-curvature
arcs. Arcs that come within `robot_radius + margin` of an obstacle point inside the
horizon (or the minimum look-ahead distance, for slow candidates) are rejected;
among the rest we pick the one closest to the user's command, with a small reward for clearance so the robot drifts away from walls smoothly
instead of scraping along them.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AssistParams:
    robot_radius: float = 0.11      # TurtleBot3 Burger footprint radius [m]
    margin: float = 0.07            # extra clearance kept around the robot [m]
    horizon: float = 1.5            # how far ahead each arc is simulated [s]
    min_lookahead: float = 0.35     # ...but never less than this distance [m], so that
                                    # crawling slowly cannot make a collision course look safe
    dt: float = 0.1                 # rollout step [s]
    # Each wheel motor is limited to 6.67 rad/s (0.22 m/s at the rim), so driving
    # and turning share one budget: |v| + |w| * separation / 2 <= 0.22.
    wheel_radius: float = 0.033
    # What diff_drive_controller uses to turn (v, w) into wheel speeds: 0.16 m times
    # the wheel_separation_multiplier (1.112) in config/ros2control.yml.
    wheel_separation: float = 0.16 * 1.112
    max_wheel_speed: float = 6.67   # [rad/s]
    max_v: float = 0.22
    max_w: float = 2.47             # turning in place at full wheel speed
    w_offsets: tuple = (0.0, -0.3, 0.3, -0.6, 0.6, -1.0, 1.0, -1.5, 1.5, -2.2, 2.2)
    v_scales: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    # Cost weights
    k_v: float = 4.0                # deviation from the user's speed (per m/s)
    k_w: float = 1.0                # deviation from the user's turn rate (per rad/s)
    k_clear: float = 0.4            # reward for clearance (per m, capped at clear_cap)
    clear_cap: float = 0.5


def rollout(v, w, horizon, dt):
    """Poses (x, y) along the arc driven by constant (v, w), starting at the origin facing +x."""
    t = np.arange(dt, horizon + 1e-9, dt)
    if abs(w) < 1e-6:
        return np.stack([v * t, np.zeros_like(t)], axis=1)
    th = w * t
    return np.stack([v / w * np.sin(th), v / w * (1 - np.cos(th))], axis=1)


def limit_wheels(v, w, p=AssistParams()):
    """Scale (v, w) down, keeping the curvature, until neither wheel exceeds its top speed."""
    rim = p.max_wheel_speed * p.wheel_radius
    need = abs(v) + abs(w) * p.wheel_separation / 2
    if need <= rim:
        return v, w
    return v * rim / need, w * rim / need


def clearance(path, obstacles):
    """Distance from each path point to the nearest obstacle point."""
    if len(obstacles) == 0:
        return np.full(len(path), np.inf)
    d = np.linalg.norm(path[:, None, :] - obstacles[None, :, :], axis=2)
    return d.min(axis=1)


def correct(v, w, obstacles, p=AssistParams()):
    """Return the safe command closest to (v, w) and how much it differs (0 = untouched, 1 = fully overridden).

    obstacles: (N, 2) array of obstacle points in the robot frame (x forward, y left).
    """
    if abs(v) < 1e-3 and abs(w) < 1e-3:
        return 0.0, 0.0, 0.0
    # A command the wheels cannot execute would be clipped per wheel by the motors,
    # bending the arc away from the one checked below. Make it feasible first.
    v, w = limit_wheels(v, w, p)

    safe_dist = p.robot_radius + p.margin
    # Only points the robot could reach in the horizon matter; this keeps the
    # distance matrix small.
    reach = max(p.max_v * p.horizon, p.min_lookahead) + safe_dist
    if len(obstacles):
        obstacles = obstacles[np.linalg.norm(obstacles, axis=1) < reach]

    # Turning in place never moves the footprint, so it is always allowed.
    best, best_cost = (0.0, float(np.clip(w, -p.max_w, p.max_w))), None
    for vs in p.v_scales:
        for dw in p.w_offsets:
            cv, cw = limit_wheels(v * vs, float(np.clip(w + dw, -p.max_w, p.max_w)), p)
            if abs(cv) < 1e-3:
                cost = p.k_v * abs(v) + p.k_w * abs(cw - w)
            else:
                duration = max(p.horizon, p.min_lookahead / abs(cv))
                path = rollout(cv, cw, duration, p.dt * duration / p.horizon)
                c = clearance(path, obstacles)
                if c.min() < safe_dist:
                    continue
                cost = (p.k_v * abs(cv - v) + p.k_w * abs(cw - w)
                        - p.k_clear * min(c.min() - safe_dist, p.clear_cap))
            if best_cost is None or cost < best_cost:
                best, best_cost = (cv, cw), cost

    # Normalised size of the change, for display / reports.
    dv = abs(best[0] - v) / p.max_v
    dw = abs(best[1] - w) / p.max_w
    return best[0], best[1], float(min(1.0, max(dv, dw)))


def scan_to_points(ranges, angle_min, angle_increment, range_max, x_off=0.0, y_off=0.0, yaw_off=0.0):
    """Convert a LaserScan's ranges to (N, 2) points in the robot frame."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_increment * np.arange(len(r)) + yaw_off
    ok = np.isfinite(r) & (r > 0.05) & (r < range_max)
    r, a = r[ok], a[ok]
    return np.stack([r * np.cos(a) + x_off, r * np.sin(a) + y_off], axis=1)
