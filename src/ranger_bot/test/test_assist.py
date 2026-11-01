import math
import numpy as np

from ranger_bot.assist import correct, rollout, scan_to_points


def wall_ahead(x=0.4, half_width=1.0):
    ys = np.linspace(-half_width, half_width, 41)
    return np.stack([np.full_like(ys, x), ys], axis=1)


def test_free_space_passes_command_through():
    v, w, k = correct(0.2, 0.2, np.empty((0, 2)))
    assert (v, w, k) == (0.2, 0.2, 0.0)


def test_zero_command_stays_zero():
    assert correct(0.0, 0.0, wall_ahead()) == (0.0, 0.0, 0.0)


def test_wall_ahead_never_drives_into_it():
    v, w, k = correct(0.22, 0.0, wall_ahead(0.4))
    path = rollout(v, w, 1.5, 0.1)
    assert np.linalg.norm(path[:, None] - wall_ahead(0.4)[None], axis=2).min() >= 0.18 or abs(v) < 1e-3
    assert k > 0


def test_obstacle_on_one_side_steers_away():
    # Post slightly left of the straight-ahead line: keep going, veer right.
    post = np.array([[0.4, 0.06], [0.4, 0.10], [0.42, 0.08]])
    v, w, _ = correct(0.15, 0.0, post)
    assert v == 0.15
    assert w < 0


def test_obstacle_too_close_to_dodge_stops():
    post = np.array([[0.25, 0.08], [0.25, 0.12], [0.27, 0.10]])
    v, _, _ = correct(0.15, 0.0, post)
    assert v == 0.0


def test_turning_in_place_is_always_allowed():
    v, w, _ = correct(0.0, 1.0, wall_ahead(0.15))
    assert (v, w) == (0.0, 1.0)


def test_scan_to_points_drops_invalid_ranges():
    pts = scan_to_points([1.0, float('inf'), 0.0, 2.0], -np.pi / 2, np.pi / 2, 3.5)
    assert pts.shape == (2, 2)
    np.testing.assert_allclose(pts[0], [0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(pts[1], [-2.0, 0.0], atol=1e-9)  # 4th ray points backwards


def test_commands_respect_wheel_speed_limit():
    from ranger_bot.assist import AssistParams, limit_wheels
    p = AssistParams()
    for v, w in [(0.2, 1.5), (0.22, 2.84), (-0.2, -1.0), (0.0, 3.0)]:
        cv, cw, _ = correct(v, w, np.empty((0, 2)))
        for wheel in (cv - cw * p.wheel_separation / 2, cv + cw * p.wheel_separation / 2):
            assert abs(wheel) / p.wheel_radius <= p.max_wheel_speed + 1e-9
    # Scaling keeps the curvature, so the robot drives the arc it asked for.
    cv, cw = limit_wheels(0.2, 1.5)
    assert abs(cw / cv - 1.5 / 0.2) < 1e-9


def test_scan_ccw_keeps_every_beam_in_place():
    from sensor_msgs.msg import LaserScan
    from ranger_bot.scan_ccw import to_ccw
    cw = LaserScan(angle_min=np.pi, angle_max=-np.pi + 2 * np.pi / 4,
                   angle_increment=-2 * np.pi / 4, range_max=3.5,
                   ranges=[1.0, 2.0, 3.0, 0.5])
    ccw = to_ccw(cw)
    assert ccw.angle_increment > 0
    a = scan_to_points(cw.ranges, cw.angle_min, cw.angle_increment, 3.5)
    b = scan_to_points(ccw.ranges, ccw.angle_min, ccw.angle_increment, 3.5)
    np.testing.assert_allclose(sorted(map(tuple, a)), sorted(map(tuple, b)), atol=1e-9)


def test_inf_beams_become_free_space():
    from sensor_msgs.msg import LaserScan
    from ranger_bot.scan_ccw import inf_to_free
    scan = LaserScan(range_min=0.12, range_max=3.5, ranges=[1.0, math.inf, 3.6, 3.5, math.nan])
    out = inf_to_free(scan)
    assert out.range_max == 4.0
    assert list(out.ranges[:4]) == [1.0, 3.5, 3.5, 3.5]
    assert math.isnan(out.ranges[4])   # NaN stays invalid


def test_keepout_mask_covers_low_rocks_only():
    from ranger_bot.keepout_mask import LIDAR_HEIGHT, parse_rocks, render_mask
    world = ('Rock {\n  translation 2 1 0\n  name "low"\n  scale 1.5\n}\n'
             'Rock {\n  translation -3 -3 0\n  name "tall"\n  type "flat"\n  scale 4\n}\n')
    rocks = parse_rocks(world)
    assert [r[0] for r in rocks] == ['low', 'tall']
    low, tall = rocks
    assert low[4] < LIDAR_HEIGHT < tall[4]
    grid = (-4.0, -4.5, 0.05, 160, 170)            # origin x, y, resolution, width, height
    rows = render_mask([low], clearance=0.10, grid=grid)
    assert (len(rows), len(rows[0])) == (170, 160)

    def cell(x, y):
        return rows[170 - 1 - int((y + 4.5) / 0.05)][int((x + 4.0) / 0.05)]
    assert cell(2.0, 1.0) == 0                     # rock centre is keepout
    assert cell(2.0 + low[3] + 0.05, 1.0) == 0     # still inside radius + clearance
    assert cell(2.0 + low[3] + 0.2, 1.0) == 205    # "unknown" (no information) beyond it
    assert cell(-3.0, -3.0) == 205                 # tall rock not in this mask


def _joy(pressed=(), axes=None):
    buttons = [0] * 13
    for i in pressed:
        buttons[i] = 1
    return buttons, axes or [0.0] * 8


def test_mode_buttons_switch_on_press_not_hold():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('manual')
    assert m.on_joy(*_joy([1]), 0.0) == ('shared', 'button')        # Circle
    assert m.on_joy(*_joy([1]), 0.1) == (None, '')                  # still held: no repeat
    assert m.on_joy(*_joy([2]), 0.2) == ('auto', 'button')          # Triangle
    assert m.on_joy(*_joy([3]), 0.3) == ('manual', 'button')        # Square
    assert m.source() == 'cmd_vel_joy'


STICK = [0.0, 0.6, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]                    # left stick forward


def test_takeover_needs_deadman_and_stick_held_for_a_second():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('auto', takeover_hold=1.0)
    assert m.on_joy(*_joy([], STICK), 0.0) == (None, '')             # no dead-man: ignored
    assert m.on_joy(*_joy([4], [0.0] * 8), 0.1) == (None, '')        # dead-man, stick centred
    assert m.on_joy(*_joy([4], STICK), 0.2) == (None, '')            # hold starts
    assert m.on_joy(*_joy([4], STICK), 0.9) == (None, '')            # 0.7 s: not yet
    assert m.on_joy(*_joy([4], STICK), 1.25) == ('shared', 'takeover')
    assert m.source() == 'cmd_vel_shared'


def test_takeover_hold_restarts_when_released():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('auto', takeover_hold=1.0)
    m.on_joy(*_joy([4], STICK), 0.0)
    m.on_joy(*_joy([4], STICK), 0.8)
    assert m.on_joy(*_joy([4], [0.0] * 8), 0.9) == (None, '')        # stick released
    assert m.on_joy(*_joy([4], STICK), 1.0) == (None, '')            # hold restarts at 1.0
    assert m.on_joy(*_joy([4], STICK), 1.9) == (None, '')
    assert m.on_joy(*_joy([4], STICK), 2.0) == ('shared', 'takeover')


def test_no_takeover_outside_auto():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('manual')
    assert m.on_joy(*_joy([4], STICK), 0.0) == (None, '')
    assert m.on_joy(*_joy([4], STICK), 5.0) == (None, '')            # normal driving


def test_set_mode_validates():
    import pytest
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('manual')
    assert m.set_mode(' AUTO ') is True and m.source() == 'cmd_vel_auto'
    assert m.set_mode('auto') is False
    with pytest.raises(ValueError):
        m.set_mode('turbo')


def test_halt_while_driving_in_shared_holds_until_deadman_repressed():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('shared')
    m.on_joy(*_joy([4], STICK), 0.0)                                  # driving: L1 + stick
    assert m.passes('cmd_vel_shared')
    assert m.halt() is False and m.mode == 'shared'                   # voice "stop"
    assert not m.passes('cmd_vel_shared')
    assert m.on_joy(*_joy([4], STICK), 0.5) == (None, '')            # still holding: stays halted
    assert not m.passes('cmd_vel_shared')
    assert m.on_joy(*_joy([], STICK), 1.0) == (None, '')             # release L1
    assert m.on_joy(*_joy([4], STICK), 1.5) == (None, 'resume')      # press again
    assert m.passes('cmd_vel_shared')


def test_halt_from_auto_drops_to_manual_and_mode_choice_clears_it():
    from ranger_bot.mode_manager import ModeLogic
    m = ModeLogic('auto')
    assert m.halt() is True and m.mode == 'manual'                    # caller cancels Nav2
    assert not m.passes('cmd_vel_joy')
    assert m.on_joy(*_joy([3]), 0.0) == ('manual', 'button')         # Square, same mode: clears
    assert m.passes('cmd_vel_joy')
    m.halt()
    assert m.set_mode('manual') is True and m.passes('cmd_vel_joy')   # /ranger/set_mode clears


def test_voice_interpret_and_grammar():
    from ranger_bot.tasks import PLACES
    from ranger_bot.voice_control import grammar, interpret
    targets = {**PLACES, 'deer': (4.0, 5.5, 0.6)}
    assert interpret('auto', targets) == ('mode', 'auto')
    assert interpret('automatic', targets) == ('mode', 'auto')
    assert interpret('go towards the deer', targets) == ('task', 'go towards the deer')
    assert interpret('[unk] go near the pond', targets) == ('task', 'go near the pond')
    assert interpret('check the center', targets) == ('task', 'check the center')
    assert interpret('stop', targets) == ('task', 'stop')
    assert interpret('[unk]', targets)[0] is None
    assert interpret('go to the [unk]', targets)[0] is None
    phrases = grammar(targets)
    assert 'go towards the deer' in phrases and 'go near the pond' in phrases
    assert 'go to the center' in phrases and 'go to the centre' not in phrases   # US model
    assert phrases[-1] == '[unk]'


def test_patrol_route_helpers():
    from ranger_bot.patrol import parse_route, rotate_to_nearest
    route = parse_route('a:0,0; b:10,0; c:10,10;')
    assert route[1] == ('b', 10.0, 0.0)
    assert [n for n, _, _ in rotate_to_nearest(route, 9.0, 1.0)] == ['b', 'c', 'a']
    assert rotate_to_nearest([], 0, 0) == []


def test_haptics_proximity():
    import math
    from ranger_bot.haptics import clearance, proximity_intensity
    pts = [(0.5, 0.0), (0.2, 0.5), (-0.3, 0.0)]                  # ahead, off to the side, behind
    assert abs(clearance(pts, 0.2) - (0.5 - 0.08)) < 1e-9        # only the point in the path
    assert abs(clearance(pts, -0.2) - (0.3 - 0.08)) < 1e-9       # reversing: the one behind
    assert clearance(pts, 0.0) == math.inf                       # turning in place: no warning
    assert proximity_intensity(1.0) == 0.0
    assert proximity_intensity(0.05) == 1.0
    assert 0.25 < proximity_intensity(0.2) < 1.0


def test_task_parsing_and_animals():
    from ranger_bot.tasks import PLACES, parse_animals, parse_task
    world = 'Deer {\n  translation 4 5.5 0\n  rotation 0 0 1 0.4\n  name "deer"\n}\n' \
            'Cat {\n  translation 5 -9.5 0\n  name "cat"\n}\n'
    animals = parse_animals(world)
    assert animals == {'deer': (4.0, 5.5, 0.6), 'cat': (5.0, -9.5, 0.25)}
    targets = {**PLACES, **animals}
    assert parse_task('go towards the deer', targets) == ('goto', 'deer')
    assert parse_task('Go near the pond!', targets) == ('goto', 'lake')
    assert parse_task('go home', targets) == ('goto', 'home')
    assert parse_task('patrol', targets) == ('patrol', None)
    assert parse_task('stop now', targets) == ('stop', None)
    assert parse_task('go to the unicorn', targets)[0] is None


def test_approach_spot_faces_target_from_robot_side():
    import math
    from ranger_bot.tasks import approach_candidates
    spots = approach_candidates((4.0, 5.5, 0.6), robot=(0.0, 5.5), standoff=1.0)
    x, y, yaw = spots[0]
    assert abs(x - (4.0 - 1.6)) < 1e-9 and abs(y - 5.5) < 1e-9   # on the robot's side
    assert abs(yaw) < 1e-9                                        # facing the deer (+x)
    assert all(abs(math.hypot(sx - 4.0, sy - 5.5) - 1.6) < 1e-9 for sx, sy, _ in spots)
    assert len(spots) == 12
    assert approach_candidates((0.0, 12.5, 0.0), robot=(0.0, 0.0))[0][:2] == (0.0, 12.5)


def test_announcer_phrases_and_speaker_report():
    from ranger_bot.announcer import mode_phrase, speaker_report, spoken_status
    assert spoken_status('Going to the deer (standing at 3.1, 4.2)') == 'Going to the deer'
    assert spoken_status('Arrived at the deer') == 'Arrived at the deer'
    assert spoken_status('Stopped - press L1/R1 again (or pick a mode) to drive') == 'Stopped'
    assert spoken_status('Not understood: no known place or animal in "x"').startswith('Sorry')
    assert spoken_status('Task to the deer cancelled') is None
    assert mode_phrase('auto') == 'Auto mode' and mode_phrase('weird') is None
    r = speaker_report(0x4F)
    assert len(r) == 32 and r[0] == 0x05 and r[1] == 0x80 and r[22] == 0x4F
    assert r[4] == r[5] == 0 and r[6:9] == bytes(3)          # rumble and LED not touched


def test_speed_levels_dpad_and_words():
    import pytest
    from ranger_bot.mode_manager import Speed
    sp = Speed()
    assert sp.fraction == 1.0
    assert sp.command('faster') is False                       # already at the top
    assert sp.command('slower') is True and sp.fraction == 0.75
    assert sp.command('min') is True and sp.fraction == 0.25
    assert sp.command('slower') is False and sp.fraction == 0.25
    assert sp.command('max') is True and sp.fraction == 1.0
    assert sp.on_dpad(-1.0) == 'slower'                         # press down
    assert sp.on_dpad(-1.0) is None                             # held: one step only
    assert sp.on_dpad(0.0) is None
    assert sp.on_dpad(1.0) == 'faster'
    with pytest.raises(ValueError):
        sp.command('warp')


def test_voice_speed_words():
    from ranger_bot.tasks import PLACES
    from ranger_bot.voice_control import grammar, interpret
    assert interpret('slow down', PLACES) == ('speed', 'slower')
    assert interpret('max speed', PLACES) == ('speed', 'max')
    assert 'speed up' in grammar(PLACES)


def test_vision_bearing_matches_known_animal():
    import math
    from ranger_bot.vision_dataset import CAM_W, bearing_of_pixel, inside, match_animal
    assert bearing_of_pixel(CAM_W / 2) == 0.0
    assert bearing_of_pixel(0) > 0.5                        # left edge: ~+0.6 rad (FOV 1.2)
    animals = {'deer': (5.0, 0.0, 0.6), 'cow': (5.0, 2.0, 0.9), 'fox': (-5.0, 0.0, 0.35)}
    pose = (0.0, 0.0, 0.0)                                  # at origin, facing +x
    c = CAM_W / 2
    assert match_animal(c - 40, c + 40, pose, animals) == 'deer'
    fx = c / math.tan(0.6)
    u_cow = c - fx * math.tan(math.atan2(2.0, 5.0 - 0.03))
    assert match_animal(u_cow - 30, u_cow + 30, pose, animals) == 'cow'   # left of centre
    assert match_animal(0, 25, pose, animals) is None                     # nothing out there
    assert match_animal(c - 40, c + 40, (0.0, 0.0, math.pi), animals) == 'fox'
    assert inside((10, 10, 20, 20), (0, 0, 100, 100)) and not inside((0, 0, 100, 100), (10, 10, 20, 20))


def test_sighting_needs_repeat_frames_then_cools_down():
    from ranger_bot.animal_detector import Confirmer, bearing
    c = Confirmer(window=5, confirm=3, cooldown=60.0)
    assert c.update(['deer'], 0.0) == []
    assert c.update([], 0.3) == []                          # a gap is fine
    assert c.update(['deer'], 0.6) == []
    assert c.update(['deer', 'fox'], 0.9) == ['deer']       # 3 of the last 5
    assert c.update(['deer'], 1.2) == []                    # cooling down
    assert c.update(['fox'], 1.5) == []                     # fox only 2 of 5 so far
    assert c.update(['fox'], 1.8) == ['fox']
    for t in range(5):
        c.update([], 2.0 + t)
    assert c.update(['deer'], 70.0) == [] and c.update(['deer'], 70.3) == []
    assert c.update(['deer'], 70.6) == ['deer']             # after the cooldown
    assert bearing(320, 640) == 0.0 and bearing(0, 640) > 0
