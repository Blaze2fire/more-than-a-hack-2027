"""Features and predictors. Everything takes a Match and returns per-frame codes (H/A/D/X).

Two predictors:
  predict_baseline  the organisers' reference (nearest player + majority vote), for comparison.
  predict_hmm       Option A: evidence for team control and for dead ball, decoded with Viterbi
                    over three states so that possessions come out as intervals, not frames.
"""
import numpy as np

from .common import A, D, FPS, H, X

DEFAULTS = dict(
    # team-control evidence
    k_team=1.0,       # weight of the per-frame log-likelihood ratio
    s_margin=2.0,     # m: scale of the distance margin between nearest home and away player
    r_near=3.0,       # m: evidence fades when nobody is this close to the ball
    v_gate=12.0,      # m/s: evidence fades when the ball moves faster than this (a pass in flight)
    # dead-ball evidence (log-odds of dead vs alive, per frame). PLAYER KINEMATICS ONLY: the
    # external games have no ball during stoppages, so any ball-based dead-ball cue would be
    # tuned against a ball track we invented ourselves (audit finding, see RESEARCH_LOG E006).
    b_dead=-0.30,     # bias
    b_slow=0.25,      # per m/s that mean player speed (1 s) is below v_ref
    b_slow_long=0.0,  # same for mean player speed over 6 s
    v_ref=2.0,        # m/s
    # switching costs
    c_team=12.0,      # H <-> A
    c_dead=10.0,      # alive -> D
    c_live=10.0,      # D -> alive
    min_spell=0,      # frames; shorter H/A spells are absorbed by a neighbouring live spell
    decouple=0,       # 1: team evidence cannot move the alive-vs-dead decision (see predict_hmm)
)


def features(m):
    """Per-frame evidence inputs. Computed once per match; no tunable parameters in here."""
    n = m.n
    bx, by = m.ball[:, 0], m.ball[:, 1]
    vis = ~np.isnan(bx)
    d = np.hypot(m.pos[:, :, 0] - bx[:, None], m.pos[:, :, 1] - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    d_home = d[:, m.team == 0].min(axis=1)
    d_away = d[:, m.team == 1].min(axis=1)

    lag = 3
    vb = np.full(n, np.nan, dtype=np.float32)
    step = m.ball[2 * lag:] - m.ball[:-2 * lag]
    vb[lag:-lag] = np.hypot(step[:, 0], step[:, 1]) * (FPS / (2.0 * lag))

    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(m.spd)).sum(axis=1)
        mspd = np.where(cnt > 0, np.nansum(m.spd, axis=1) / np.maximum(cnt, 1), np.nan)
    mspd_long = _smooth_nan(mspd, 3 * FPS)
    mspd = _smooth_nan(mspd, FPS)

    length, width = m.pitch
    oob = np.maximum.reduce([-bx, bx - length, -by, by - width])
    in_block = np.zeros(n, dtype=bool)
    for b0, b1 in m.blocks:
        in_block[b0:b1] = True
    return dict(n=n, vis=vis, d_home=d_home, d_away=d_away, vb=vb, mspd=mspd,
                mspd_long=mspd_long, oob=oob,
                blocks=m.blocks, in_block=in_block)


def _smooth_nan(x, half):
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


# -- baseline ------------------------------------------------------------------------------------

def predict_baseline(F, window=50):
    """Nearest tracked player to the ball, then a centred majority vote. Mirrors starter/baseline.py."""
    codes = np.where(~F["vis"], -1, np.where(F["d_home"] <= F["d_away"], H, A))
    codes = np.where(np.isinf(np.minimum(F["d_home"], F["d_away"])), -1, codes)
    if window <= 0:
        out = np.where(codes == H, H, np.where(codes == A, A, D))
    else:
        hc = np.concatenate(([0], np.cumsum(codes == H)))
        ac = np.concatenate(([0], np.cumsum(codes == A)))
        i = np.arange(F["n"])
        lo, hi = np.maximum(0, i - window), np.minimum(F["n"], i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        out = np.where((h + a) > 0, np.where(h >= a, H, A), D)
    return _mask(out, F)


def _mask(out, F):
    out = out.astype(np.int8)
    out[~F["in_block"]] = X
    return out


# -- Option A: evidence + Viterbi ----------------------------------------------------------------

def evidence(F, cfg):
    """(team_llr, dead_logodds): per-frame evidence for home-over-away and for dead-over-alive."""
    vis = F["vis"]
    with np.errstate(invalid="ignore", over="ignore"):
        margin = F["d_away"] - F["d_home"]
        margin = np.where(np.isfinite(margin), margin, np.sign(margin) * 50.0)
        near = np.minimum(F["d_home"], F["d_away"])
        w_near = np.exp(-(near / cfg["r_near"]) ** 2)
        vb = np.where(np.isnan(F["vb"]), 0.0, F["vb"])
        w_speed = 1.0 / (1.0 + (vb / cfg["v_gate"]) ** 2)
        llr = cfg["k_team"] * np.tanh(margin / cfg["s_margin"]) * w_near * w_speed
    llr = np.where(vis & np.isfinite(llr), llr, 0.0)

    mspd = np.where(np.isnan(F["mspd"]), cfg["v_ref"], F["mspd"])
    mlong = np.where(np.isnan(F["mspd_long"]), cfg["v_ref"], F["mspd_long"])
    dead = (cfg["b_dead"] + cfg["b_slow"] * (cfg["v_ref"] - mspd)
            + cfg["b_slow_long"] * (cfg["v_ref"] - mlong))
    return llr, dead


def viterbi3(s_h, s_a, s_d, c_team, c_dead, c_live):
    """Best H/A/D path for per-frame scores and switching costs. Plain-float loop, three states."""
    n = len(s_h)
    back = np.empty((n, 3), dtype=np.int8)
    vh, va, vd = float(s_h[0]), float(s_a[0]), float(s_d[0])
    sh, sa, sd = s_h.tolist(), s_a.tolist(), s_d.tolist()
    bk = back.tolist()
    for t in range(1, n):
        # into H
        a1, a2, a3 = vh, va - c_team, vd - c_live
        if a1 >= a2 and a1 >= a3:
            nh, bh = a1, 0
        elif a2 >= a3:
            nh, bh = a2, 1
        else:
            nh, bh = a3, 2
        # into A
        a1, a2, a3 = vh - c_team, va, vd - c_live
        if a2 >= a1 and a2 >= a3:
            na, ba = a2, 1
        elif a1 >= a3:
            na, ba = a1, 0
        else:
            na, ba = a3, 2
        # into D
        a1, a2, a3 = vh - c_dead, va - c_dead, vd
        if a3 >= a1 and a3 >= a2:
            nd, bd = a3, 2
        elif a1 >= a2:
            nd, bd = a1, 0
        else:
            nd, bd = a2, 1
        vh, va, vd = nh + sh[t], na + sa[t], nd + sd[t]
        row = bk[t]
        row[0], row[1], row[2] = bh, ba, bd
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax([vh, va, vd]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = bk[t][state]
    return path


def absorb_short(codes, min_len):
    """Absorb H/A spells shorter than min_len into a neighbouring LIVE spell.

    Preference: the live spell before it, else the live spell after it. A short live spell with
    dead ball on both sides is left alone: play that happened is never relabelled as dead ball.
    (The first version copied the preceding label unconditionally, which turned a short
    possession straight after a restart into D. Audit finding, fixed in E006.)
    """
    if min_len <= 0:
        return codes
    out = codes.copy()
    breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(out)]))
    for i in range(len(bounds) - 1):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if out[s] == D or e - s >= min_len:
            continue
        prev_k = out[s - 1] if s > 0 else D
        next_k = out[e] if e < len(out) else D
        if prev_k != D:
            out[s:e] = prev_k
        elif next_k != D:
            out[s:e] = next_k
    return out


def predict_hmm(F, cfg=None, extra=None):
    """Option A. `extra` = optional (team_llr_add, dead_add) arrays, e.g. Smart Tagging priors."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    llr, dead = evidence(F, cfg)
    if extra is not None:
        llr = llr + extra[0]
        dead = dead + extra[1]
    out = np.full(F["n"], X, dtype=np.int8)
    for b0, b1 in F["blocks"]:
        l, dd = llr[b0:b1], dead[b0:b1]
        if cfg["decouple"]:
            # Best live state always scores 0, so only `dd` decides alive vs dead.
            s_h, s_a = np.minimum(0.0, l), np.minimum(0.0, -l)
        else:
            # Symmetric split: the favoured team scores +|l|/2, so strong team evidence also
            # counts as evidence that the ball is alive.
            s_h, s_a = 0.5 * l, -0.5 * l
        path = viterbi3(s_h, s_a, dd, cfg["c_team"], cfg["c_dead"], cfg["c_live"])
        out[b0:b1] = absorb_short(path, int(cfg["min_spell"]))
    return out
