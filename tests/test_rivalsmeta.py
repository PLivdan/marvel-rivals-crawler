import json
import pathlib

import rivalsmeta

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeClient:
    """Stands in for fetcher.RivalsMetaClient, returning canned fixture data."""

    def __init__(self, json_by_path=None, text_by_path=None):
        self.json_by_path = json_by_path or {}
        self.text_by_path = text_by_path or {}
        self.calls = []

    def get_json(self, path, params=None):
        self.calls.append((path, params))
        return self.json_by_path[path]

    def get_text(self, path, params=None):
        self.calls.append((path, params))
        return self.text_by_path[path]


def test_is_diamond_plus_boundaries():
    assert rivalsmeta.is_diamond_plus(12) is False
    assert rivalsmeta.is_diamond_plus(13) is True
    assert rivalsmeta.is_diamond_plus(22) is True
    assert rivalsmeta.is_diamond_plus(23) is True
    assert rivalsmeta.is_diamond_plus(None) is False


def test_get_global_leaderboard_parses_devalue_payload():
    raw_text = (FIXTURES / "leaderboard_payload.json").read_text()
    client = FakeClient(text_by_path={"/leaderboard/_payload.json": raw_text})
    board = rivalsmeta.get_global_leaderboard(client)
    assert len(board["players"]) == 500
    assert board["players"][0]["name"] == "KovaaksKid2008"


def test_resolve_current_season_reads_rank_game_id_from_leaderboard():
    raw_text = (FIXTURES / "leaderboard_payload.json").read_text()
    client = FakeClient(text_by_path={"/leaderboard/_payload.json": raw_text})
    assert rivalsmeta.resolve_current_season(client) == 19


def test_get_hero_leaderboard_returns_plain_json():
    payload = load("hero_leaderboard.json")
    client = FakeClient(json_by_path={"/api/hero-leaderboard/1047": payload})
    result = rivalsmeta.get_hero_leaderboard(client, 1047)
    assert result["_id"] == 1047
    assert len(result["players"]) > 0


def test_get_player_returns_profile():
    payload = load("player_public.json")
    client = FakeClient(json_by_path={"/api/player/457877313": payload})
    result = rivalsmeta.get_player(client, 457877313, season=19)
    assert result["player"]["_id"] == 457877313


def test_get_player_match_history_page_passes_expected_params():
    payload = load("player_match_history_page.json")
    client = FakeClient(json_by_path={"/api/player-match-history/457877313": payload})
    result = rivalsmeta.get_player_match_history_page(client, 457877313, skip=20, season=19)
    assert result == payload
    path, params = client.calls[0]
    assert params == {"skip": 20, "game_mode_id": rivalsmeta.MATCHMODE_COMPETITIVE, "hero_id": 0, "season": 19}


def test_get_match_detail_returns_full_match():
    payload = load("match_detail.json")
    client = FakeClient(json_by_path={"/api/matches/m1": payload})
    result = rivalsmeta.get_match_detail(client, "m1")
    assert len(result["match_players"]) == 12


def test_current_season_rank_extracts_matching_season_blob():
    profile = load("player_public.json")
    blob = rivalsmeta.current_season_rank(profile, season=19)
    assert blob["level"] == 22
    assert round(blob["rank_score"], 2) == 5132.57


def test_current_season_rank_returns_none_for_unknown_season():
    profile = load("player_public.json")
    assert rivalsmeta.current_season_rank(profile, season=999) is None
