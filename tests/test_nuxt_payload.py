import json
import pathlib

from nuxt_payload import resolve_payload

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_resolve_payload_reconstructs_leaderboard():
    raw = (FIXTURES / "leaderboard_payload.json").read_text()
    data = resolve_payload(raw)
    assert data["device"] == "1"
    assert data["season"] == "last"
    assert len(data["players"]) == 500
    first = data["players"][0]
    assert first["name"] == "KovaaksKid2008"
    assert first["uid"] == "1822797559"
    assert first["rank"]["rank_game_id"] == 19


def test_resolve_payload_is_pure_json_compatible():
    # sanity: the raw fixture must itself be valid JSON (an array),
    # confirming this is devalue-over-JSON, not a bespoke text format.
    raw = (FIXTURES / "leaderboard_payload.json").read_text()
    arr = json.loads(raw)
    assert isinstance(arr, list)
