"""Populate the hero_info / teamups / teamup_heroes reference tables.

Unlike the crawler, this reads the site's *client bundle*, not the JSON API.
rivalsmeta serves hero and team-up names nowhere in /api/* — the ids that come
back on match endpoints are opaque, and the names live in a `JSON.parse(`...`)`
literal inside one of the Nuxt chunks that /characters loads. So the flow is:
fetch /characters, walk its /_nuxt/*.js chunks, and take the one containing the
hero database.

The chunk filename is a content hash that changes on every site rebuild, so it
is discovered at run time rather than hardcoded — a pinned URL would 404 the
first time they deploy. This is reference data that changes at most once a
season, so it is a manual script rather than anything the crawl calls.

Usage: python3 refresh_reference.py --db-path data/rivals.db
"""

import argparse
import json
import re
import sys
import time

import requests

import db
import fetcher

CHARACTERS_URL = "https://rivalsmeta.com/characters"
CHUNK_URL = "https://rivalsmeta.com/_nuxt/{name}"

# The hero database is the only JSON.parse literal in the bundle that starts
# with a hero_id, which is what makes it identifiable without executing any JS.
BUNDLE_MARKER = 'JSON.parse(`[{"hero_id":'

# Courtesy delay between chunk fetches. The crawler may well be running against
# the same origin while this script walks the bundle.
CHUNK_DELAY_SECONDS = 0.5


def _untemplate(text):
    """Undo JS *template-literal* escaping only.

    The literal is wrapped in backticks, so the bundler escapes backslash,
    backtick and `$`. JSON's own escapes (\\n, \\", \\uXXXX) must survive
    untouched for json.loads, which is why this cannot be a blanket
    unicode_escape / replace('\\\\', '\\') pass — that would turn the `\\"` in
    a name like JONATHAN \\"JOHNNY\\" STORM into a bare quote and truncate the
    string mid-record.
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in ("\\", "`", "$"):
                out.append(nxt)
            else:
                out.append(ch)
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def discover_bundle(session):
    """Return the JS source of the chunk holding the hero database."""
    page = session.get(CHARACTERS_URL, timeout=30)
    page.raise_for_status()
    names = sorted(set(re.findall(r"/_nuxt/([A-Za-z0-9._-]+\.js)", page.text)))
    if not names:
        raise RuntimeError("no /_nuxt/*.js chunks referenced by /characters")
    for name in names:
        resp = session.get(CHUNK_URL.format(name=name), timeout=30)
        if resp.status_code != 200:
            continue
        if BUNDLE_MARKER in resp.text:
            print(f"hero database found in /_nuxt/{name}", file=sys.stderr)
            return resp.text
        time.sleep(CHUNK_DELAY_SECONDS)
    raise RuntimeError(f"no chunk among {len(names)} contained {BUNDLE_MARKER!r}")


def parse_heroes(js_text):
    start = js_text.index(BUNDLE_MARKER) + len("JSON.parse(`")
    end = js_text.index("`)", start)
    return json.loads(_untemplate(js_text[start:end]))


def load(conn, heroes):
    """Upsert heroes and their team-ups. Returns (hero_count, teamup_count)."""
    for h in heroes:
        db.upsert(
            conn,
            "hero_info",
            ["hero_id"],
            {
                "hero_id": h["hero_id"],
                "name": h["name"],
                "role": h.get("role"),
                "class": h.get("class"),
                "difficulty": h.get("difficulty"),
                "gender": h.get("gender"),
                "attack_method": h.get("attackMethod"),
                "internal_name": h.get("internalName"),
                "real_name": h.get("realName"),
                "refreshed_at": db.now(),
            },
        )

    # A team-up is listed on the record of every hero it involves, so the same
    # id arrives once per member. Collapse to one definition before writing.
    teamups = {}
    for h in heroes:
        for t in h.get("Teamup") or []:
            teamups[t["id"]] = t

    for tid, t in teamups.items():
        db.upsert(
            conn,
            "teamups",
            ["teamup_id"],
            {
                "teamup_id": tid,
                "name": t["name"],
                "anchor_hero_id": t.get("anchor"),
                "start_sub_season": t.get("startSubSeason"),
                "end_sub_season": t.get("endSubSeason"),
                "description": t.get("text"),
                "refreshed_at": db.now(),
            },
        )
        # Membership is rewritten rather than upserted: a reworked team-up can
        # drop a hero, and a plain upsert would leave that stale edge behind.
        conn.execute("DELETE FROM teamup_heroes WHERE teamup_id=?", (tid,))
        for hero_id in t.get("heroes") or []:
            db.upsert(
                conn,
                "teamup_heroes",
                ["teamup_id", "hero_id"],
                {
                    "teamup_id": tid,
                    "hero_id": hero_id,
                    "is_anchor": 1 if hero_id == t.get("anchor") else 0,
                },
            )
    return len(heroes), len(teamups)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default="data/rivals.db")
    args = ap.parse_args(argv)

    session = requests.Session()
    session.headers["User-Agent"] = fetcher.RivalsMetaClient.USER_AGENT

    heroes = parse_heroes(discover_bundle(session))
    conn = db.connect(args.db_path)
    db.init_schema(conn)
    with conn:
        n_heroes, n_teamups = load(conn, heroes)
    print(f"loaded {n_heroes} heroes, {n_teamups} team-ups", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
