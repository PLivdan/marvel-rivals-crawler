import json

from nuxt_payload import resolve_payload

MATCHMODE_COMPETITIVE = 2
RANK_LEVEL_DIAMOND_MIN = 13


def is_diamond_plus(level):
    return level is not None and level >= RANK_LEVEL_DIAMOND_MIN


def get_global_leaderboard(client):
    text = client.get_text("/leaderboard/_payload.json")
    return resolve_payload(text)


def get_hero_leaderboard(client, hero_id):
    return client.get_json(f"/api/hero-leaderboard/{hero_id}", params={"device": 1, "season": "last"})


def get_player(client, uid, season):
    return client.get_json(f"/api/player/{uid}", params={"season": season})


def get_player_match_history_page(client, uid, skip, season, game_mode_id=MATCHMODE_COMPETITIVE):
    return client.get_json(
        f"/api/player-match-history/{uid}",
        params={"skip": skip, "game_mode_id": game_mode_id, "hero_id": 0, "season": season},
    )


def get_match_detail(client, match_uid):
    return client.get_json(f"/api/matches/{match_uid}")


def resolve_current_season(client):
    board = get_global_leaderboard(client)
    return board["players"][0]["rank"]["rank_game_id"]


def current_season_rank(profile, season):
    """Extract the {level, rank_score, ...} blob for the given season from
    a player profile's info.rank_game_<1001000+season> field, which is a
    JSON-encoded string, not a nested object."""
    info = profile.get("player", {}).get("info", {})
    raw = info.get(f"rank_game_{1001000 + season}")
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed.get("rank_game")
