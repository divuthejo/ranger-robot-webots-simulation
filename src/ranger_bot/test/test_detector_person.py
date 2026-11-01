"""Person-box handling in animal_detector (plausible_person, resolve_people)."""
from ranger_bot.animal_detector import implied_height, iou, plausible_person, resolve_people

INTRUDER = (401, 163, 435, 267)                     # 9 m, person model 0.88


def test_real_intruder_boxes_kept():
    for box in [(541, 162, 565, 263), INTRUDER, (441, 141, 484, 271), (576, 54, 640, 290),
                (93, 184, 119, 256)]:                         # 2026-11-01 run, 0.88
        assert 1.3 < implied_height(box) < 1.8 and plausible_person(box) == 'full', box


def test_animals_rejected_as_people():
    assert not plausible_person((386, 243, 414, 268))     # cat at 8 m
    assert not plausible_person((279, 226, 300, 263))     # distant deer, 0.60 'person'
    assert not plausible_person((143, 184, 234, 260))     # horse, implied 1.38 m but wide
    assert not plausible_person((63, 68, 228, 324))       # cow from behind (aspect 0.64)


# 2026-11-01: a horse seen from behind (0.85) raised the intruder alarm; a deer facing the
# camera and slivers of horse legs at the image edge also came back as 'person'.
HORSE_REAR = [(584, 147, 640, 280), (451, 115, 522, 291), (385, 113, 456, 292),
              (424, 99, 505, 298), (448, 77, 541, 306), (477, 42, 592, 318)]
DEER_FRONT = [(223, 174, 260, 268), (221, 161, 265, 271), (193, 144, 250, 277),
              (165, 120, 238, 285)]
EDGE_SLIVERS = [(0, 15, 24, 284), (0, 69, 28, 297), (610, 87, 640, 290)]
HORSE_LEGS_CUT = [(494, 2, 637, 336), (522, 0, 640, 330), (581, 7, 640, 315), (569, 2, 640, 315)]


def test_end_on_animals_and_slivers_rejected():
    for box in HORSE_REAR + DEER_FRONT + EDGE_SLIVERS:
        assert plausible_person(box) is None, box


def test_head_cut_boxes_only_continue_a_person():
    for box in HORSE_LEGS_CUT:
        assert plausible_person(box) == 'cut', box
        dets = [(box, 'person', 0.9)]
        assert resolve_people([], dets)[1] == []                       # nobody seen before
        assert resolve_people([], dets, recent_full=True)[1] == dets   # following a person
    # a whole person in the same frame also vouches for a cut box
    both = [(INTRUDER, 'person', 0.9), (HORSE_LEGS_CUT[0], 'person', 0.5)]
    assert len(resolve_people([], both)[1]) == 2


def test_person_beats_the_cow_label_on_the_intruder():
    cow_on_person = ((399, 160, 437, 268), 'cow', 0.8)
    real_cow = ((100, 180, 190, 262), 'cow', 0.95)
    animals, people, saw_full = resolve_people(
        [cow_on_person, real_cow],
        [(INTRUDER, 'person', 0.88), ((100, 182, 188, 260), 'person', 0.7)])
    assert [p[0] for p in people] == [INTRUDER] and saw_full
    assert animals == [real_cow]
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
