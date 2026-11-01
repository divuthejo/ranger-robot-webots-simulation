"""BatteryLogic and IntruderLogic policies (no ROS)."""

import math

from ranger_bot.battery import BatteryLogic
from ranger_bot.intruder import IntruderLogic, nearest_place

AWAY, HOME = (5.0, 5.0), (0.1, 0.0)


def run(b, seconds, **kw):
    acts = []
    for k in range(int(seconds)):
        acts += b.step(1.0, kw.get('speed', 0.15), kw.get('pose', AWAY), kw.get('moving', True),
                       kw.get('mode', 'auto'), kw.get('t0', 0) + k, kw.get('patrolling', True))
    return acts


def kinds(acts):
    return [a[0] for a in acts]


def test_full_battery_lasts_about_half_an_hour_patrolling():
    b = BatteryLogic(level=100.0, low=0.0, warn=-1)
    run(b, 1800)
    assert 0 < b.level < 15


def test_low_in_auto_goes_home_once_and_resumes_patrol_after_charge():
    b = BatteryLogic(level=21.0)
    acts = run(b, 120)
    assert kinds(acts).count('go_home') == 1 and b.returning and b.resume_patrol
    acts = run(b, 400, pose=HOME, moving=False, speed=0.0)
    assert ('say', 'Charging') in acts and kinds(acts).count('patrol') == 1
    assert b.level >= 95 and not b.returning


def test_low_in_manual_only_reminds_never_drives():
    b = BatteryLogic(level=19.0)
    acts = run(b, 130, mode='manual')
    assert 'go_home' not in kinds(acts) and kinds(acts).count('say') == 3   # t=0, 60, 120


def test_takeover_during_return_does_not_resume_patrol_later():
    b = BatteryLogic(level=20.5)
    run(b, 30)                                   # auto: sent home
    assert b.returning
    run(b, 5, mode='manual')                     # ranger takes over
    acts = run(b, 400, pose=HOME, moving=False, speed=0.0, mode='manual')
    assert 'patrol' not in kinds(acts)


def test_warn_once_and_empty_halts_once():
    b = BatteryLogic(level=26.0, low=-1)
    acts = run(b, 600, mode='manual')
    assert kinds(acts).count('rumble') == 1
    b = BatteryLogic(level=0.5)
    acts = run(b, 20, mode='manual')
    assert kinds(acts).count('halt') == 1


def test_no_charging_while_driving_over_the_dock():
    b = BatteryLogic(level=50.0)
    run(b, 10, pose=HOME, moving=True)
    assert b.level < 50.0 and not b.charging


def person(x, y, age=0.5):
    return {'person': {'cls': 'person', 'x': x, 'y': y, 'age_s': age}}


def test_intruder_alert_face_lost_resume():
    lg = IntruderLogic(lost_after=20)
    acts = lg.feed(person(3.0, 3.0), (0.0, 0.0, 0.0), 'auto', 100.0)
    assert acts[0] == ('alert', 3.0, 3.0)
    face = [a for a in acts if a[0] == 'face']
    assert face and abs(face[0][1] - math.pi / 4) < 1e-6
    assert lg.feed(person(3.0, 3.0), (0.0, 0.0, math.pi / 4), 'auto', 101.0) == []
    assert lg.feed({}, (0, 0, 0), 'auto', 110.0) == []          # not lost yet
    acts = lg.feed({}, (0, 0, 0), 'auto', 125.0)
    assert acts == [('lost',), ('resume_patrol',)]


def test_intruder_in_manual_alerts_but_never_turns_or_resumes():
    lg = IntruderLogic()
    acts = lg.feed(person(3.0, -3.0), (0.0, 0.0, 0.0), 'manual', 0.0)
    assert [a[0] for a in acts] == ['alert']
    acts = lg.feed({}, (0, 0, 0), 'manual', 100.0)
    assert acts == [('lost',)]


def test_stale_person_track_is_not_an_intruder():
    lg = IntruderLogic()
    assert lg.feed(person(1, 1, age=30.0), (0, 0, 0), 'auto', 0.0) == []


def test_nearest_place():
    assert nearest_place(7.0, 6.0) == 'lake' and nearest_place(-8.0, 2.0) == 'forest'
