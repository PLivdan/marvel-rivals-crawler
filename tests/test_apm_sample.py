import db
from apm import sample


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def add_match(conn, uid, *, duration=700.0, timestamp=1_787_000_000,
              camp0_wins=True, n_players=12, draw=False,
              missing_score=False, missing_playtime=False):
    """Insert one synthetic match with a full 6v6 roster."""
    conn.execute(
        "INSERT INTO matches (match_uid, match_time_stamp, match_play_duration) "
        "VALUES (?,?,?)", (uid, timestamp, duration))
    for i in range(n_players):
        camp = 0 if i < 6 else 1
        won = (camp == 0) == camp0_wins
        is_win = 2 if (draw and i == 0) else int(won)
        conn.execute(
            "INSERT INTO match_players (match_uid, player_uid, camp, cur_hero_id,"
            " is_win, add_score, new_score) VALUES (?,?,?,?,?,?,?)",
            (uid, 1000 + i, camp, 1011 + i, is_win,
             None if missing_score else 10.0, None if missing_score else 4500.0))
        if not missing_playtime:
            conn.execute(
                "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
                " play_time) VALUES (?,?,?,?)", (uid, 1000 + i, 1011 + i, 600.0))
    conn.commit()


def test_clean_match_is_retained():
    conn = make_conn()
    add_match(conn, "m1")
    frame, report = sample.build_sample(conn)
    assert list(frame["match_uid"]) == ["m1"]
    assert frame.loc[0, "camp0_win"] == 1
    assert report.remaining == 1


def test_outcome_flips_with_the_winning_camp():
    conn = make_conn()
    add_match(conn, "m1", camp0_wins=False)
    frame, _ = sample.build_sample(conn)
    assert frame.loc[0, "camp0_win"] == 0


def test_each_exclusion_rule_drops_its_match_and_is_counted():
    conn = make_conn()
    add_match(conn, "keep")
    add_match(conn, "short", duration=100.0)
    add_match(conn, "draw", draw=True)
    add_match(conn, "noscore", missing_score=True)
    add_match(conn, "notime", missing_playtime=True)
    add_match(conn, "roster", n_players=10)
    frame, report = sample.build_sample(conn, forfeit_floor_seconds=240)
    assert list(frame["match_uid"]) == ["keep"]
    assert report.starting == 6
    assert report.dropped_roster == 1
    assert report.dropped_draw == 1
    assert report.dropped_missing_score == 1
    assert report.dropped_missing_playtime == 1
    assert report.dropped_short == 1
    assert report.remaining == 1


def test_forfeit_floor_is_configurable():
    conn = make_conn()
    add_match(conn, "m1", duration=200.0)
    assert sample.build_sample(conn, forfeit_floor_seconds=0)[1].remaining == 1
    assert sample.build_sample(conn, forfeit_floor_seconds=240)[1].remaining == 0


def test_temporal_holdout_splits_on_time_not_at_random():
    conn = make_conn()
    day = 86_400
    for i in range(10):
        add_match(conn, f"m{i}", timestamp=1_787_000_000 + i * day)
    frame, _ = sample.build_sample(conn)
    train, test = sample.temporal_holdout(frame, holdout_days=3)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(frame)
    assert len(test) == 3
