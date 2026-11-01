"""Person-box handling in animal_detector (plausible_person, resolve_people)."""
from ranger_bot.animal_detector import implied_height, iou, plausible_person, resolve_people

INTRUDER = (401, 163, 435, 267)                     # 9 m, person model 0.88


def test_real_intruder_boxes_kept():
    for box in [(541, 162, 565, 263), INTRUDER, (441, 141, 484, 271), (576, 54, 640, 290)]:
        assert 1.3 < implied_height(box) < 1.8 and plausible_person(box), box


def test_animals_rejected_as_people():
    assert not plausible_person((386, 243, 414, 268))     # cat at 8 m
    assert not plausible_person((279, 226, 300, 263))     # distant deer, 0.60 'person'
    assert not plausible_person((143, 184, 234, 260))     # horse, implied 1.38 m but wide
    assert not plausible_person((63, 68, 228, 324))       # cow from behind (aspect 0.64)


def test_person_beats_the_cow_label_on_the_intruder():
    cow_on_person = ((399, 160, 437, 268), 'cow', 0.8)
    real_cow = ((100, 180, 190, 262), 'cow', 0.95)
    animals, people = resolve_people([cow_on_person, real_cow],
                                     [(INTRUDER, 'person', 0.88), ((100, 182, 188, 260), 'person', 0.7)])
    assert [p[0] for p in people] == [INTRUDER]
    assert animals == [real_cow]
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
