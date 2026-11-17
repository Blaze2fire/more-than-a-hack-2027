"""4-state restart-aware Viterbi decoder & conservative spell post-processor.

States per frame inside a contiguous half block:
  0: H   (Home live possession)
  1: A   (Away live possession)
  2: D_H (Dead ball whose most recent live owner was Home)
  3: D_A (Dead ball whose most recent live owner was Away)
Collapsed output maps {D_H, D_A} -> D (2), preserving 3-class submission codes.

Key structural advantages over 3-state Viterbi:
- Different transition penalties for same-team restart (D_H -> H, e.g. free kick
  won by H) vs opposite-team restart (D_H -> A, e.g. throw-in / goal kick).
- Minimum duration hygiene / dead-dip absorption: prevents brief 1-2s uncertainty
  dips inside a live spell from splitting a GT spell in two (fatal for spellF1 IoU),
  and suppresses spurious tiny live blips inside stoppages.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from solution.common import A, D, H, X

INF = 1e9


def viterbi4_block(
    s_h: np.ndarray,
    s_a: np.ndarray,
    s_d: np.ndarray,
    c_team: float,
    c_stop: float,
    c_res_same: float,
    c_res_flip: float,
    d_bias: float = 0.0,
    sw_signal: Optional[np.ndarray] = None,
    c_sw_gain: float = 0.0,
) -> np.ndarray:
    """Optimal 4-state Viterbi decoding over one contiguous half block."""
    n = len(s_h)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    # Convert to Python lists for tight scalar C-speed DP loop (~0.06s for 150k frames).
    sh = s_h.tolist()
    sa = s_a.tolist()
    sd = (s_d + d_bias).tolist()

    d0 = float(sh[0])
    d1 = float(sa[0])
    d2 = float(sd[0])
    d3 = float(sd[0])

    bp0 = [0] * n
    bp1 = [1] * n
    bp2 = [2] * n
    bp3 = [3] * n

    ct = float(c_team)
    cs = float(c_stop)
    rs = float(c_res_same)
    rf = float(c_res_flip)
    cg = float(c_sw_gain)
    use_dyn = sw_signal is not None and cg > 0.0
    if use_dyn:
        ct_arr = np.maximum(0.35 * ct, ct - cg * np.asarray(sw_signal, dtype=np.float64)).tolist()
    else:
        ct_arr = None

    for t in range(1, n):
        ct_t = ct_arr[t] if ct_arr is not None else ct

        # Next = H (from H, A, DH, DA)
        c00 = d0
        c10 = d1 - ct_t
        c20 = d2 - rs
        c30 = d3 - rf
        if c00 >= c10:
            best0, prev0 = c00, 0
        else:
            best0, prev0 = c10, 1
        if c20 > best0:
            best0, prev0 = c20, 2
        if c30 > best0:
            best0, prev0 = c30, 3
        nd0 = best0 + sh[t]

        # Next = A (from H, A, DH, DA)
        c01 = d0 - ct_t
        c11 = d1
        c21 = d2 - rf
        c31 = d3 - rs
        if c11 >= c01:
            best1, prev1 = c11, 1
        else:
            best1, prev1 = c01, 0
        if c21 > best1:
            best1, prev1 = c21, 2
        if c31 > best1:
            best1, prev1 = c31, 3
        nd1 = best1 + sa[t]

        # Next = DH (from H -> DH or stay DH -> DH)
        c02 = d0 - cs
        if d2 >= c02:
            best2, prev2 = d2, 2
        else:
            best2, prev2 = c02, 0
        nd2 = best2 + sd[t]

        # Next = DA (from A -> DA or stay DA -> DA)
        c13 = d1 - cs
        if d3 >= c13:
            best3, prev3 = d3, 3
        else:
            best3, prev3 = c13, 1
        nd3 = best3 + sd[t]

        d0, d1, d2, d3 = nd0, nd1, nd2, nd3
        bp0[t] = prev0
        bp1[t] = prev1
        bp2[t] = prev2
        bp3[t] = prev3

    # Backtrack
    state = 0
    best_end = d0
    if d1 > best_end:
        best_end, state = d1, 1
    if d2 > best_end:
        best_end, state = d2, 2
    if d3 > best_end:
        state = 3

    out = [0] * n
    bps = (bp0, bp1, bp2, bp3)
    for t in range(n - 1, -1, -1):
        out[t] = D if state >= 2 else state
        state = bps[state][t]
    return np.asarray(out, dtype=np.int8)


def repair_spells_block(
    codes: np.ndarray,
    logp: Optional[np.ndarray] = None,
    min_dead: int = 25,
    min_live_blip: int = 18,
    min_turn: int = 12,
    snap_win: int = 0,
    snap_sig: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Targeted cleanup of spell-splitting micro-artifacts & symmetric boundary snapping."""
    c = codes.copy()
    n = len(c)
    if n == 0:
        return c

    def _runs(arr: np.ndarray):
        p = np.concatenate([[-1], arr, [-1]])
        ch = np.where(p[1:] != p[:-1])[0]
        return ch[:-1].tolist(), ch[1:].tolist(), arr[ch[:-1]].tolist()

    # 1. Absorb tiny isolated dead dips flanked by live spells.
    if min_dead > 0:
        starts, ends, vals = _runs(c)
        for i, (s, e, v) in enumerate(zip(starts, ends, vals)):
            L = e - s
            if v == D and L < min_dead and 0 < i < len(vals) - 1:
                lv, rv = vals[i - 1], vals[i + 1]
                if lv in (H, A) and rv in (H, A):
                    if lv == rv:
                        c[s:e] = lv
                    elif logp is not None:
                        c[s:e] = H if logp[s:e, H].mean() >= logp[s:e, A].mean() else A

    # 2. Suppress tiny live blips strictly inside dead ball.
    if min_live_blip > 0:
        starts, ends, vals = _runs(c)
        for i, (s, e, v) in enumerate(zip(starts, ends, vals)):
            L = e - s
            if v in (H, A) and L < min_live_blip and 0 < i < len(vals) - 1:
                if vals[i - 1] == D and vals[i + 1] == D:
                    c[s:e] = D

    # 3. Absorb ultra-short H <-> A flips flanked by the SAME opposite team.
    if min_turn > 0:
        starts, ends, vals = _runs(c)
        for i, (s, e, v) in enumerate(zip(starts, ends, vals)):
            L = e - s
            if v in (H, A) and L < min_turn and 0 < i < len(vals) - 1:
                if vals[i - 1] == vals[i + 1] and vals[i - 1] in (H, A):
                    c[s:e] = vals[i - 1]

    # 4. Symmetric boundary snapping on local changepoint peak.
    if snap_win > 0 and snap_sig is not None:
        starts, ends, vals = _runs(c)
        min_keep = max(min_turn, 6)
        for i in range(len(vals) - 1):
            lv, rv = vals[i], vals[i + 1]
            if lv in (H, A) and rv in (H, A) and lv != rv:
                t0 = ends[i]
                lo = max(starts[i] + min_keep, t0 - snap_win)
                hi = min(ends[i + 1] - min_keep, t0 + snap_win)
                if lo <= hi:
                    t_best = lo + int(np.argmax(snap_sig[lo : hi + 1]))
                    if snap_sig[t_best] > snap_sig[t0] + 0.05 and snap_sig[t_best] > 0.30:
                        if t_best < t0:
                            c[t_best:t0] = rv
                        elif t_best > t0:
                            c[t0:t_best] = lv

    return c


def _symmetric_prob_changepoint(logp_blk: np.ndarray, win: int = 10) -> np.ndarray:
    """Symmetric H<->A step contrast |fut_mean(p_h) - past_mean(p_h)| inside a block."""
    n = len(logp_blk)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    ph = np.exp(logp_blk[:, H].astype(np.float64))
    pa = np.exp(logp_blk[:, A].astype(np.float64))
    p_h_live = ph / np.maximum(ph + pa, 1e-6)
    cs = np.concatenate([[0.0], np.cumsum(p_h_live)])
    idx = np.arange(n)
    p_lo = np.maximum(0, idx - win)
    f_hi = np.minimum(n, idx + win)
    p_mean = (cs[idx] - cs[p_lo]) / np.maximum(idx - p_lo, 1)
    f_mean = (cs[f_hi] - cs[idx]) / np.maximum(f_hi - idx, 1)
    diff = np.round(np.clip(np.abs(f_mean - p_mean) * 1.5, 0.0, 1.0) * 65536.0) / 65536.0
    return diff.astype(np.float32)


DECODE4_DEFAULTS = dict(
    w=1.0,
    d_bias=0.0,
    c_team=3.5,
    c_sw_gain=0.0,
    c_stop=14.0,
    c_res_same=10.0,
    c_res_flip=12.0,
    min_dead=35,
    min_live_blip=20,
    min_turn=14,
    snap_win=0,
    ownership_pass=False,
    ownership_cost_scale=0.5,
)


def ownership_pass(logp, codes, blocks, c_team, sw_signal=None, c_sw_gain=0.0):
    """Re-decode live spans with a frozen dead mask (Ming's two-pass idea).

    Dynamic programming sees whole live spans and retains short spells when
    their total evidence pays the switch cost. It never joins spans across D/X.
    """
    out = codes.copy()
    for start, stop in blocks:
        live = (codes[start:stop] == H) | (codes[start:stop] == A)
        edges = np.diff(np.r_[False, live, False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            lo, hi = start + int(lo), start + int(hi)
            scores = logp[lo:hi, :2].astype(np.float64)
            n = hi - lo
            dp = scores[0].copy()
            back = np.empty((n, 2), dtype=np.int8)
            for t in range(1, n):
                cost = float(c_team)
                if sw_signal is not None and c_sw_gain > 0:
                    cost = max(0.35 * cost, cost - c_sw_gain * float(sw_signal[lo + t]))
                stay_h, flip_h = dp[H], dp[A] - cost
                stay_a, flip_a = dp[A], dp[H] - cost
                back[t, H] = H if stay_h >= flip_h else A
                back[t, A] = A if stay_a >= flip_a else H
                dp = np.array([max(stay_h, flip_h), max(stay_a, flip_a)]) + scores[t]
            state = int(np.argmax(dp))
            for t in range(n - 1, -1, -1):
                out[lo + t] = state
                if t:
                    state = int(back[t, state])
    return out


def decode4(
    logp: np.ndarray,
    blocks: List[Tuple[int, int]],
    cfg: Optional[Dict[str, float]] = None,
    sw_signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode (n, 3) log-probabilities over match half blocks using viterbi4 + spell repair."""
    p = dict(DECODE4_DEFAULTS, **(cfg or {}))
    if p.get("freeze_dead_from_gbdt", False):
        raise ValueError("freeze_dead_from_gbdt requires PipelineV4.predict or an explicit reference mask")
    c_sw_gain = float(p.get("c_sw_gain", 0.0))
    snap_win = int(p.get("snap_win", 0))
    out = np.full(len(logp), X, dtype=np.int8)
    s = float(p["w"]) * logp
    for b0, b1 in blocks:
        blk_sw = None
        if c_sw_gain > 0.0 or snap_win > 0:
            step_sig = _symmetric_prob_changepoint(logp[b0:b1], win=10)
            if sw_signal is not None:
                blk_sw = 0.55 * np.asarray(sw_signal[b0:b1], dtype=np.float32) + 0.45 * step_sig
            else:
                blk_sw = step_sig
        path = viterbi4_block(
            s[b0:b1, H],
            s[b0:b1, A],
            s[b0:b1, D],
            c_team=float(p["c_team"]),
            c_stop=float(p["c_stop"]),
            c_res_same=float(p["c_res_same"]),
            c_res_flip=float(p["c_res_flip"]),
            d_bias=float(p["d_bias"]),
            sw_signal=blk_sw,
            c_sw_gain=c_sw_gain,
        )
        out[b0:b1] = repair_spells_block(
            path,
            logp=logp[b0:b1],
            min_dead=int(p["min_dead"]),
            min_live_blip=int(p["min_live_blip"]),
            min_turn=int(p["min_turn"]),
            snap_win=snap_win,
            snap_sig=blk_sw,
        )
    if p.get("ownership_pass", False):
        scale = float(p["ownership_cost_scale"])
        if not 0.0 <= scale <= 1.0:
            raise ValueError("ownership_cost_scale must be between zero and one")
        out = ownership_pass(s, out, blocks, float(p["c_team"]) * scale,
                             sw_signal=sw_signal, c_sw_gain=c_sw_gain * scale)
    return out

