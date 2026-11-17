"""Ball-gated dead-ball timing: snap a predicted stoppage onset to an observed line crossing.

The player-only dead head decides WHETHER a stoppage happens; this step only decides WHEN it
started, and only when the ball is observed leaving the pitch with continuity. It never creates or
deletes a dead spell, never merges two stoppages and never changes which team is in possession.

Parameters were fixed from the four DFL training fixtures before any held-out scoring:
55% of true onsets have an observed exit in [-3 s, +1 s], at a median 2 frames before the
provider's dead flag.
"""
import numpy as np

from solution.common import D

DEFAULTS = dict(margin=0.3, back=100, fwd=25, delay=2, cont=10, max_step=3.0)


def _exits(ball, pitch, margin, cont, max_step):
    """Frames t where the observed ball is outside the lines and was observed inside within
    `cont` frames before, with every observed step on the way under `max_step` metres."""
    L, W = pitch
    x, y = ball[:, 0], ball[:, 1]
    obs = np.isfinite(x)
    out = obs & ((x < -margin) | (x > L + margin) | (y < -margin) | (y > W + margin))
    inside = obs & ~out
    n = len(ball)
    ok = np.zeros(n, dtype=bool)
    for t in np.where(out)[0]:
        lo = max(0, t - cont)
        prev_in = np.where(inside[lo:t])[0]
        if len(prev_in) == 0:
            continue
        seg = np.arange(lo + prev_in[-1], t + 1)
        seg = seg[obs[seg]]
        steps = np.hypot(np.diff(x[seg]), np.diff(y[seg])) / np.maximum(np.diff(seg), 1)
        if np.all(steps < max_step):
            ok[t] = True
    return ok


def snap_onsets(codes, ball, pitch, blocks, cfg=None):
    """Return a copy of `codes` (H/A/D) with dead-spell onsets moved to observed line exits."""
    p = dict(DEFAULTS, **(cfg or {}))
    out = codes.copy()
    exits = _exits(ball, pitch, p["margin"], int(p["cont"]), p["max_step"])
    for b0, b1 in blocks:
        seg = out[b0:b1]
        dead = seg == D
        starts = np.where(dead[1:] & ~dead[:-1])[0] + 1
        for s in starts:
            e = s
            while e < len(seg) and seg[e] == D:
                e += 1
            prev_d_end = 0
            back = np.where(seg[:s] == D)[0]
            if len(back):
                prev_d_end = back[-1] + 1
            lo, hi = max(prev_d_end + 1, s - int(p["back"])), min(e - 1, s + int(p["fwd"]))
            cand = np.where(exits[b0 + lo:b0 + hi + 1])[0]
            if len(cand) == 0:
                continue
            new = min(lo + cand[0] + int(p["delay"]), e - 1)
            if new <= prev_d_end or new == s:
                continue
            if new < s:
                seg[new:s] = D
            else:
                seg[s:new] = seg[s - 1]
        out[b0:b1] = seg
    return out
