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
) -> np.ndarray:
    """Optimal 4-state Viterbi decoding over one contiguous half block."""
    n = len(s_h)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    # Convert to Python lists for tight scalar C-speed DP loop (~0.06s for 150k frames).
    sh = s_h.tolist()
    sa = s_a.tolist()
    sd = (s_d + d_bias).tolist()

    # Transition matrix T[prev][cur]
    # 0=H, 1=A, 2=DH, 3=DA
    # Dead states never switch directly between DH <-> DA (cost INF).
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

    for t in range(1, n):
        # Next = H (from H, A, DH, DA)
        c00 = d0
        c10 = d1 - ct
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
        c01 = d0 - ct
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
    ball_acc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Targeted cleanup of spell-splitting micro-artifacts inside one contiguous half."""
    c = codes.copy()
    n = len(c)
    if n == 0:
        return c

    def _runs(arr: np.ndarray):
        p = np.concatenate([[-1], arr, [-1]])
        ch = np.where(p[1:] != p[:-1])[0]
        return ch[:-1].tolist(), ch[1:].tolist(), arr[ch[:-1]].tolist()

    # 1. Absorb tiny isolated dead dips strictly flanked by the same/live team
    #    (stops H -> D(short) -> H from slicing one spell in half!).
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
                        # Fill with higher average live team log-prob
                        c[s:e] = H if logp[s:e, H].mean() >= logp[s:e, A].mean() else A

    # 2. Suppress tiny live blips strictly inside dead ball (D -> live(tiny) -> D).
    if min_live_blip > 0:
        starts, ends, vals = _runs(c)
        for i, (s, e, v) in enumerate(zip(starts, ends, vals)):
            L = e - s
            if v in (H, A) and L < min_live_blip and 0 < i < len(vals) - 1:
                if vals[i - 1] == D and vals[i + 1] == D:
                    c[s:e] = D

    # 3. Absorb ultra-short H <-> A flips flanked by the SAME opposite team (H -> A(tiny) -> H).
    if min_turn > 0:
        starts, ends, vals = _runs(c)
        for i, (s, e, v) in enumerate(zip(starts, ends, vals)):
            L = e - s
            if v in (H, A) and L < min_turn and 0 < i < len(vals) - 1:
                if vals[i - 1] == vals[i + 1] and vals[i - 1] in (H, A):
                    c[s:e] = vals[i - 1]

    return c


DECODE4_DEFAULTS = dict(
    w=1.0,
    d_bias=0.0,
    c_team=3.5,
    c_stop=14.0,
    c_res_same=10.0,
    c_res_flip=7.0,
    min_dead=35,
    min_live_blip=20,
    min_turn=14,
)


def decode4(
    logp: np.ndarray,
    blocks: List[Tuple[int, int]],
    cfg: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Decode (n, 3) log-probabilities over match half blocks using viterbi4 + spell repair."""
    p = dict(DECODE4_DEFAULTS, **(cfg or {}))
    out = np.full(len(logp), X, dtype=np.int8)
    s = float(p["w"]) * logp
    for b0, b1 in blocks:
        path = viterbi4_block(
            s[b0:b1, H],
            s[b0:b1, A],
            s[b0:b1, D],
            c_team=float(p["c_team"]),
            c_stop=float(p["c_stop"]),
            c_res_same=float(p["c_res_same"]),
            c_res_flip=float(p["c_res_flip"]),
            d_bias=float(p["d_bias"]),
        )
        out[b0:b1] = repair_spells_block(
            path,
            logp=logp[b0:b1],
            min_dead=int(p["min_dead"]),
            min_live_blip=int(p["min_live_blip"]),
            min_turn=int(p["min_turn"]),
        )
    return out

