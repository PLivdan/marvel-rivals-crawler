import pytest

import db
import crawler
import fetcher
import rivalsmeta

pytestmark = pytest.mark.live


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(str(tmp_path / "live_smoke.db"))
    db.init_schema(connection)
    return connection


@pytest.fixture
def client():
    return fetcher.RivalsMetaClient()


def test_resolve_current_season_against_live_site(client):
    season = rivalsmeta.resolve_current_season(client)
    assert isinstance(season, int) and season > 0


def test_crawl_a_known_public_diamond_plus_player(conn, client):
    # A verified-public player used throughout development of this crawler.
    uid = 457877313
    season = rivalsmeta.resolve_current_season(client)
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    status = crawler.crawl_player(conn, client, uid, season)

    assert status in ("done", "skipped_floor")  # rank may have changed since verification
    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    if status == "done":
        assert n_matches >= 1
        n_bans = conn.execute("SELECT COUNT(*) FROM match_bans").fetchone()[0]
        assert n_bans >= 0  # some game modes may have zero bans; just confirm no crash


def test_hero_leaderboard_shape_matches_expectations(client):
    board = rivalsmeta.get_hero_leaderboard(client, hero_id=1047)  # Jeff The Land Shark
    assert board["_id"] == 1047
    assert len(board["players"]) > 0
