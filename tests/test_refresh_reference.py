import json

import db
import refresh_reference as R

HEROES = [
    {"hero_id": 1024, "name": "Hela", "role": "Damage",
     "Teamup": [{"id": 1024001, "name": "HEL TENDRILS", "anchor": 1024, "heroes": [1024, 1016]}]},
    {"hero_id": 1016, "name": "Loki", "role": "Support", "Teamup": []},
    {"hero_id": 1067, "name": "Gorr The God Butcher", "role": "Damage", "Teamup": []},
]
RAGNAROK = {"id": 1024003, "name": "RAGNAROK", "anchor": 1024, "heroes": [1024, 1067]}


def _members(conn, tid):
    return sorted(r[0] for r in conn.execute("SELECT hero_id FROM teamup_heroes WHERE teamup_id=?", (tid,)))


def test_supplement_adds_team_up_missing_from_site(tmp_path):
    conn = db.connect(str(tmp_path / "r.db")); db.init_schema(conn)
    with conn:
        _, n = R.load(conn, HEROES, [RAGNAROK])
    assert n == 2
    assert _members(conn, 1024003) == [1024, 1067]
    assert conn.execute("SELECT anchor_hero_id FROM teamups WHERE teamup_id=1024003").fetchone()[0] == 1024
    assert conn.execute("SELECT is_anchor FROM teamup_heroes WHERE teamup_id=1024003 AND hero_id=1024").fetchone()[0] == 1


def test_site_record_wins_once_it_lists_the_same_members(tmp_path):
    heroes = json.loads(json.dumps(HEROES))
    heroes[2]["Teamup"] = [{"id": 1024009, "name": "RAGNAROK", "anchor": 1024, "heroes": [1067, 1024]}]
    conn = db.connect(str(tmp_path / "r.db")); db.init_schema(conn)
    with conn:
        _, n = R.load(conn, heroes, [RAGNAROK])
    assert n == 2
    assert conn.execute("SELECT COUNT(*) FROM teamups WHERE teamup_id=1024003").fetchone()[0] == 0


def test_shipped_supplement_is_valid():
    sup = json.load(open(R.SUPPLEMENT_PATH))["teamups"]
    assert {t["name"] for t in sup} == {"RAGNAROK", "HIVE MIND"}
    for t in sup:
        assert t["anchor"] in t["heroes"] and 1067 in t["heroes"]
        assert t["id"] // 1000 == t["anchor"]
