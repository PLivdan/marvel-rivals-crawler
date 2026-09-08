"""Result persistence and formatted tables.

Results are written back into the crawler's database so a coefficient can be
traced to the exact sample and specification that produced it months later.
"""

import numpy as np
import pandas as pd


def to_probability_points(beta):
    """Convert a log-odds coefficient to win-probability points.

    The logit marginal effect is beta * p * (1-p), which at a balanced match
    (p = 0.5) is beta / 4. Reported in points, so multiply by 100.
    """
    return beta / 4.0 * 100.0


def record_run(conn, spec, attribution_rule, forfeit_floor, n_matches,
               n_players, exclusions, git_commit):
    cur = conn.execute(
        "INSERT INTO apm_runs (spec, attribution_rule, forfeit_floor,"
        " n_matches, n_players, exclusions, git_commit) VALUES (?,?,?,?,?,?,?)",
        (spec, attribution_rule, forfeit_floor, n_matches, n_players,
         exclusions, git_commit),
    )
    conn.commit()
    return cur.lastrowid


def write_hero_effects(conn, run_id, fit, intervals, adjusted_p):
    errors = np.sqrt(np.diag(fit.hero_cov))
    for i, hero_id in enumerate(fit.hero_ids):
        low, high = intervals.get(hero_id, (None, None))
        conn.execute(
            "INSERT OR REPLACE INTO apm_hero_effects (run_id, hero_id,"
            " effect_logodds, std_error, ci_low, ci_high, p_adjusted)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, hero_id, float(fit.hero_effects[hero_id]),
             float(errors[i]), low, high, adjusted_p.get(hero_id)),
        )
    conn.commit()


def hero_table(conn, run_id):
    rows = conn.execute(
        "SELECT e.hero_id, h.name, h.role, e.effect_logodds, e.std_error,"
        "       e.ci_low, e.ci_high, e.p_adjusted "
        "FROM apm_hero_effects e LEFT JOIN hero_info h ON h.hero_id=e.hero_id "
        "WHERE e.run_id=? ORDER BY e.effect_logodds DESC", (run_id,)
    ).fetchall()
    frame = pd.DataFrame(rows, columns=[
        "hero_id", "name", "role", "effect_logodds", "std_error",
        "ci_low", "ci_high", "p_adjusted"])
    frame["effect_pp"] = frame["effect_logodds"].apply(to_probability_points)
    return frame
