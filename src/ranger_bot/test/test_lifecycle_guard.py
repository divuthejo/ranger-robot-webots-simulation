from ranger_bot.lifecycle_guard import ACTIVATE, CONFIGURE, NAV2_NODES, transitions_for


def test_transitions_to_active():
    assert transitions_for('unconfigured') == [CONFIGURE, ACTIVATE]
    assert transitions_for('inactive') == [ACTIVATE]
    assert transitions_for('active') == [] and transitions_for('finalized') == []


def test_guards_localization_and_navigator():
    assert {'map_server', 'amcl', 'bt_navigator', 'rock_mask_server'} <= set(NAV2_NODES)
