"""tasks.Follower: when to re-approach a followed animal, when to give up; voice phrases."""
from ranger_bot.tasks import PLACES, Follower, parse_task
from ranger_bot.voice_control import grammar, interpret


def track(x, y, age=0.5):
    return {'x': x, 'y': y, 'age_s': age}


def test_regoal_only_after_it_moved_and_not_while_driving():
    f = Follower(move=1.0, min_interval=3.0)
    f.start('deer', (4.0, 5.5), 0.0)
    assert f.update(track(4.5, 5.5), False, 5.0) is None          # moved 0.5 m: stay
    assert f.update(track(5.3, 5.5), True, 6.0) is None           # moved, but still driving
    assert f.update(track(5.3, 5.5), False, 6.5) == 'regoal'
    assert f.anchor == (5.3, 5.5)
    assert f.update(track(6.6, 5.5), False, 7.0) is None          # too soon after the last one
    assert f.update(track(6.6, 5.5), False, 9.6) == 'regoal'


def test_lost_and_timeout():
    f = Follower(lost_after=30, max_time=300)
    f.start('dog', (10, 2), 0.0)
    assert f.update(track(10, 2, age=31), False, 40) == 'lost'
    assert f.update(None, False, 40) == 'lost'
    assert f.update(track(10, 2), False, 301) == 'timeout'
    f.stop()
    assert f.update(track(10, 2), False, 302) is None


def test_follow_phrases():
    targets = dict(PLACES, deer=(4, 5.5, 0.6))
    assert parse_task('follow the deer', targets) == ('goto', 'deer')
    assert 'follow the deer' in grammar(targets)
    kind, text = interpret('follow the deer', targets)
    assert kind == 'task' and 'deer' in text


def test_look_around_before_giving_up():
    f = Follower(look_after=4.0, lost_after=30.0)
    f.start('deer', (4.0, 5.5), 0.0)
    assert f.update(track(4.0, 5.5, age=5.0), False, 6.0) == 'look'
    assert f.update(track(4.0, 5.5, age=8.0), False, 9.0) is None      # looked once already
    assert f.update(track(4.2, 5.5, age=0.5), False, 12.0) is None     # seen again
    assert f.update(track(4.2, 5.5, age=6.0), False, 18.0) == 'look'   # can look again
    assert f.update(track(4.2, 5.5, age=31.0), False, 43.0) == 'lost'


def test_find_the_intruder_parsing_and_voice():
    from ranger_bot.tasks import parse_task
    targets = dict(PLACES, deer=(4, 5.5, 0.6))
    for text in ('find the intruder', 'search for the intruder', 'follow the intruder',
                 'look for the person'):
        assert parse_task(text, targets) == ('find', 'intruder'), text
    assert 'find the intruder' in grammar(targets)
    assert interpret('find the intruder', targets)[0] == 'task'


def test_searcher_goes_to_last_seen_first_and_looks_at_every_stop():
    from ranger_bot.tasks import Searcher
    s = Searcher(PLACES, last_seen=(5.0, 5.0))
    steps = []
    while True:
        st = s.next_step()
        steps.append(st)
        if st[0] == 'done':
            break
    assert steps[0] == ('go', 'last-seen spot', (5.0, 5.0)) and steps[1] == ('look',)
    gos = [st[1] for st in steps if st[0] == 'go']
    assert gos[1:] == ['centre', 'lake', 'meadow', 'forest', 'ridge', 'home']
    assert sum(st[0] == 'look' for st in steps) == len(gos)


def test_freshest_person_track():
    from ranger_bot.tasks import freshest
    seen = {'person': {'cls': 'person', 'age_s': 90}, 'person 3': {'cls': 'person', 'age_s': 1},
            'deer': {'cls': 'deer', 'age_s': 0}}
    assert freshest(seen, 'person')[0] == 'person 3' and freshest(seen, 'fox') is None
