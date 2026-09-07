import db


def ingest_match(conn, match_detail, season, history_entry=None):
    """Store one match. `history_entry` is the corresponding entry from the
    player-match-history page that surfaced this match — it is the ONLY place
    the map id is exposed (`match_map_id`); /api/matches/{uid} carries no
    map_id/match_map_id key at all, so without it matches.map_id is always
    NULL."""
    match_uid = match_detail["match_uid"]

    db.upsert(
        conn,
        "matches",
        ["match_uid"],
        {
            "match_uid": match_uid,
            "match_time_stamp": match_detail.get("match_time_stamp"),
            "match_play_duration": match_detail.get("match_play_duration"),
            "game_mode_id": match_detail.get("game_mode_id"),
            "map_id": history_entry.get("match_map_id") if history_entry else None,
            "season": season,
            "winner_side": _winner_side(match_detail),
            "mvp_uid": match_detail.get("mvp_uid"),
            "mvp_hero_id": match_detail.get("mvp_hero_id"),
            "svp_uid": match_detail.get("svp_uid"),
            "svp_hero_id": match_detail.get("svp_hero_id"),
            "replay_id": match_detail.get("replay_id"),
            "fetched_at": db.now(),
        },
    )

    for ban in match_detail.get("dynamic_fields", {}).get("ban_pick_info", []):
        hero_id = int(ban["hero_id"])
        db.upsert(
            conn,
            "match_bans",
            ["match_uid", "round_idx", "battle_side", "hero_id"],
            {
                "match_uid": match_uid,
                "round_idx": int(ban["round_idx"]),
                "battle_side": int(ban["battle_side"]),
                "hero_id": hero_id,
                "is_pick": int(ban["is_pick"]),
            },
        )
        _record_hero_seen(conn, hero_id)

    new_player_uids = []
    for p in match_detail.get("match_players", []):
        uid = p["player_uid"]
        cur_hero_id = p.get("cur_hero_id")
        dyn = p.get("dynamic_fields", {}) or {}

        db.upsert(
            conn,
            "match_players",
            ["match_uid", "player_uid"],
            {
                "match_uid": match_uid,
                "player_uid": uid,
                "camp": p.get("camp"),
                "cur_hero_id": cur_hero_id,
                "k": p.get("k"),
                "d": p.get("d"),
                "a": p.get("a"),
                "total_hero_damage": p.get("total_hero_damage"),
                "total_hero_heal": p.get("total_hero_heal"),
                "total_damage_taken": p.get("total_damage_taken"),
                "is_win": p.get("is_win"),
                "add_score": dyn.get("add_score"),
                "new_score": dyn.get("new_score"),
                "level": dyn.get("level"),
                "new_level": dyn.get("new_level"),
                "session_hit_rate": p.get("session_hit_rate"),
            },
        )

        for hero_segment in p.get("player_heroes", []):
            hero_id = hero_segment["hero_id"]
            db.upsert(
                conn,
                "match_player_heroes",
                ["match_uid", "player_uid", "hero_id"],
                {
                    "match_uid": match_uid,
                    "player_uid": uid,
                    "hero_id": hero_id,
                    "k": hero_segment.get("k"),
                    "d": hero_segment.get("d"),
                    "a": hero_segment.get("a"),
                    "play_time": hero_segment.get("play_time"),
                },
            )
            _record_hero_seen(conn, hero_id)

        if cur_hero_id is not None:
            _record_hero_seen(conn, cur_hero_id)

        created = _upsert_discovered_player(
            conn,
            uid=uid,
            nick_name=p.get("nick_name"),
            score=dyn.get("new_score"),
            level=dyn.get("new_level"),
            discovery_hero_id=cur_hero_id,
        )
        if created:
            new_player_uids.append(uid)

    conn.commit()
    return new_player_uids


def _winner_side(match_detail):
    for p in match_detail.get("match_players", []):
        if p.get("is_win"):
            return p.get("camp")
    return None


def _record_hero_seen(conn, hero_id):
    db.upsert(conn, "heroes", ["hero_id"], {"hero_id": hero_id})


def _upsert_discovered_player(conn, uid, nick_name, score, level, discovery_hero_id):
    """Refresh a player's freshest known name/score/level. discovery_hero_id
    and crawl_status are set ONLY on first creation and never overwritten
    afterwards, so a player already tagged (e.g. from hero-leaderboard
    seeding) keeps that tag no matter how many other matches surface them."""
    existing = conn.execute("SELECT uid FROM players WHERE uid=?", (uid,)).fetchone()
    if existing is None:
        db.upsert(
            conn,
            "players",
            ["uid"],
            {
                "uid": uid,
                "nick_name": nick_name,
                "latest_known_score": score,
                "latest_known_level": level,
                "discovery_hero_id": discovery_hero_id,
                "crawl_status": "pending",
            },
        )
        return True

    db.upsert(
        conn,
        "players",
        ["uid"],
        {
            "uid": uid,
            "nick_name": nick_name,
            "latest_known_score": score,
            "latest_known_level": level,
        },
    )
    return False
