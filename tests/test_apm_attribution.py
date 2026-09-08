import pytest

import db
from apm import attribution


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def add_player(conn, match_uid, player_uid, camp, hero_playtimes):
    # match_players.match_uid references matches(match_uid) and db.connect
    # turns on foreign_keys, so a parent row must exist first (OR IGNORE
    # because add_player is called once per player on the same match_uid).
    conn.execute(
        "INSERT OR IGNORE INTO matches (match_uid) VALUES (?)", (match_uid,))
    conn.execute(
        "INSERT INTO match_players (match_uid, player_uid, camp, is_win) "
        "VALUES (?,?,?,1)", (match_uid, player_uid, camp))
    for hero_id, seconds in hero_playtimes.items():
        conn.execute(
            "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
            " play_time) VALUES (?,?,?,?)", (match_uid, player_uid, hero_id, seconds))
    conn.commit()


def test_w1_splits_a_swapping_player_across_heroes_by_time():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 300.0, 1022: 100.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W1")
    weights = dict(zip(frame["hero_id"], frame["weight"]))
    assert weights[1011] == pytest.approx(0.75)
    assert weights[1022] == pytest.approx(0.25)


def test_w2_gives_the_whole_slot_to_the_dominant_hero():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 300.0, 1022: 100.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W2")
    assert list(frame["hero_id"]) == [1011]
    assert frame.loc[0, "weight"] == pytest.approx(1.0)


def test_w2_tie_does_not_hand_the_player_two_slots():
    conn = make_conn()
    # Player 1 swapped between two heroes and played each for exactly the
    # same amount of time: a play-time tie must still collapse to one row,
    # or the camp total silently becomes 7.0 instead of 6.0.
    add_player(conn, "m1", 1, 0, {2001: 300.0, 2002: 300.0})
    for i in range(2, 7):
        add_player(conn, "m1", i, 0, {3000 + i: 300.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W2")
    tied_rows = frame[frame["hero_id"].isin({2001, 2002})]
    assert len(tied_rows) == 1
    assert tied_rows["weight"].iloc[0] == pytest.approx(1.0)
    totals = frame.groupby("camp")["weight"].sum()
    assert totals[0] == pytest.approx(6.0)


def test_weights_sum_to_six_per_team_under_both_rules():
    conn = make_conn()
    for i in range(6):
        add_player(conn, "m1", i, 0, {1011 + i: 400.0, 1040: 200.0})
    for i in range(6):
        add_player(conn, "m1", 100 + i, 1, {1030 + i: 600.0})
    for rule in attribution.RULES:
        frame = attribution.attribution_weights(conn, {"m1"}, rule)
        totals = frame.groupby("camp")["weight"].sum()
        assert totals[0] == pytest.approx(6.0)
        assert totals[1] == pytest.approx(6.0)


def test_zero_playtime_heroes_are_ignored():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 400.0, 1022: 0.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W1")
    assert set(frame["hero_id"]) == {1011}


def test_unknown_rule_is_rejected():
    conn = make_conn()
    with pytest.raises(ValueError):
        attribution.attribution_weights(conn, set(), "W9")
