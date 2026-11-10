"""Smart Tagging as soft priors on the decoder (competition match only).

No external game has these tags, so nothing here can be tuned. The weights are fixed, reasoned
values (docs/RESEARCH_LOG.md records the argument), and the stage is switched on or off as a whole.

  * A phase tagged with a team is evidence that this team has the ball during the interval.
    Attacking phases name the team in possession; DEFENDING phases name the team WITHOUT it.
  * A SET PIECES interval contains a restart: play is live by its end, and dead ball is likely
    just before it. Windows open ~3.5 s early (starter/01_smart_tagging.ipynb), so the first
    part of the window leans dead and the end leans alive.
  * Team "N/A" (all throw-ins, some others) carries no team evidence.
"""
import numpy as np

HOME_NAMES = ("barcelona",)
ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
          "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
W_PHASE = 0.15        # per-frame log-likelihood ratio added for a team-tagged phase
W_RESTART_DEAD = 0.5  # per-frame dead log-odds in the lead-in of a set-piece window
W_RESTART_LIVE = 0.5  # per-frame alive log-odds at the end of a set-piece window
LEAD_IN = 75          # frames (~3 s) of the window treated as lead-in


def _side(tag_team):
    t = (tag_team or "").strip().lower()
    if not t or t == "n/a":
        return None
    return 0 if any(name in t for name in HOME_NAMES) else 1


def priors(m, params=None):
    """(team_llr_add, dead_add) arrays over all frames."""
    params = params or {}
    w_phase = params.get("w_phase", W_PHASE)
    llr = np.zeros(m.n, dtype=np.float64)
    dead = np.zeros(m.n, dtype=np.float64)
    for ev in m.tags or []:
        code = (ev["code"] or "").strip().upper()
        s, e = max(0, ev["start_frame"]), min(m.n, ev["end_frame"])
        if e <= s:
            continue
        side = _side(ev["tags"].get("Team"))
        if code == "SET PIECES":
            lead = min(e, s + LEAD_IN)
            dead[s:lead] += W_RESTART_DEAD
            dead[lead:e] -= W_RESTART_LIVE
            if side is not None:
                llr[lead:e] += w_phase * (1 if side == 0 else -1)
        elif side is not None and code in ATTACK:
            llr[s:e] += w_phase * (1 if side == 0 else -1)
        elif side is not None and code in DEFEND:
            llr[s:e] -= w_phase * (1 if side == 0 else -1)
    # Overlapping tags are one piece of evidence, not several: cap instead of summing.
    return np.clip(llr, -w_phase, w_phase), np.clip(dead, -W_RESTART_LIVE, W_RESTART_DEAD)
