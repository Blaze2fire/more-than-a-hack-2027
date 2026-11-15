"""Stage-2 temporal sequence stackers (ball-free Dead Stacker + H<->A symmetric Team Stacker)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from solution.features_v4 import _fwd_bwd_diff_blocks, _roll_mean_blocks


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _causal_ema_blocks(arr: np.ndarray, blocks: List[Tuple[int, int]], alpha: float) -> np.ndarray:
    """Fast forward IIR exponential moving average inside each contiguous block."""
    from scipy.signal import lfilter
    out = np.zeros_like(arr, dtype=np.float32)
    b = np.array([float(alpha)], dtype=np.float64)
    a = np.array([1.0, -(1.0 - float(alpha))], dtype=np.float64)
    arr64 = np.nan_to_num(arr, nan=0.0).astype(np.float64, copy=False)
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        seg = arr64[b0:b1]
        # Initialize zi so step 0 starts at seg[0] without zero-padding transient
        zi = (1.0 - float(alpha)) * seg[0:1]
        y, _ = lfilter(b, a, seg, axis=0, zi=zi)
        out[b0:b1] = y.astype(np.float32)
    return out


def _anticausal_ema_blocks(arr: np.ndarray, blocks: List[Tuple[int, int]], alpha: float) -> np.ndarray:
    """Backward IIR exponential moving average inside each contiguous block."""
    from scipy.signal import lfilter
    out = np.zeros_like(arr, dtype=np.float32)
    b = np.array([float(alpha)], dtype=np.float64)
    a = np.array([1.0, -(1.0 - float(alpha))], dtype=np.float64)
    arr64 = np.nan_to_num(arr, nan=0.0).astype(np.float64, copy=False)
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        seg = arr64[b0:b1][::-1]
        zi = (1.0 - float(alpha)) * seg[0:1]
        y, _ = lfilter(b, a, seg, axis=0, zi=zi)
        out[b0:b1] = y[::-1].astype(np.float32)
    return out


def _since_block(cond: np.ndarray, blocks: List[Tuple[int, int]], cap: int = 300) -> np.ndarray:
    """Normalized frames [0, 1] since last True inside each contiguous block."""
    out = np.ones(len(cond), dtype=np.float32)
    inv_cap = np.float32(1.0 / float(cap))
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        c = cond[b0:b1]
        L = b1 - b0
        ar = np.arange(L, dtype=np.int32)
        idx = np.where(c, ar, -cap - 1)
        last_idx = np.maximum.accumulate(idx)
        dist = np.minimum(ar - last_idx, cap)
        out[b0:b1] = dist.astype(np.float32) * inv_cap
    return out


def _until_block(cond: np.ndarray, blocks: List[Tuple[int, int]], cap: int = 300) -> np.ndarray:
    """Normalized frames [0, 1] until next True inside each contiguous block."""
    out = np.ones(len(cond), dtype=np.float32)
    inv_cap = np.float32(1.0 / float(cap))
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        c = cond[b0:b1][::-1]
        L = b1 - b0
        ar = np.arange(L, dtype=np.int32)
        idx = np.where(c, ar, -cap - 1)
        last_idx = np.maximum.accumulate(idx)
        dist = np.minimum(ar - last_idx, cap)
        out[b0:b1] = dist[::-1].astype(np.float32) * inv_cap
    return out


def _last_val_block(
    vals: np.ndarray, valid: np.ndarray, blocks: List[Tuple[int, int]], default: float = 0.0
) -> np.ndarray:
    """Forward-fill last value where valid is True within each contiguous block."""
    out = np.full(len(vals), default, dtype=np.float32)
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        vb = valid[b0:b1]
        sb = vals[b0:b1]
        ar = np.arange(b1 - b0, dtype=np.int32)
        idx = np.where(vb, ar, -1)
        last_idx = np.maximum.accumulate(idx)
        out[b0:b1] = np.where(last_idx >= 0, sb[np.maximum(last_idx, 0)], default)
    return out


def _next_val_block(
    vals: np.ndarray, valid: np.ndarray, blocks: List[Tuple[int, int]], default: float = 0.0
) -> np.ndarray:
    """Backward-fill next value where valid is True within each contiguous block."""
    out = np.full(len(vals), default, dtype=np.float32)
    for b0, b1 in blocks:
        if b1 <= b0:
            continue
        vb = valid[b0:b1][::-1]
        sb = vals[b0:b1][::-1]
        ar = np.arange(b1 - b0, dtype=np.int32)
        idx = np.where(vb, ar, -1)
        last_idx = np.maximum.accumulate(idx)
        res = np.where(last_idx >= 0, sb[np.maximum(last_idx, 0)], default)
        out[b0:b1] = res[::-1]
    return out


def _one_sided_mean_block(
    X: np.ndarray, blocks: List[Tuple[int, int]], w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return strictly-past [t-w, t) and strictly-future (t, t+w] means inside each block."""
    past = np.zeros_like(X, dtype=np.float32)
    fut = np.zeros_like(X, dtype=np.float32)
    X64 = np.nan_to_num(X, nan=0.0).astype(np.float64, copy=False)
    for b0, b1 in blocks:
        L = b1 - b0
        if L <= 0:
            continue
        seg = X64[b0:b1]
        cs = np.zeros((L + 1, X.shape[1]), dtype=np.float64)
        cs[1:] = np.cumsum(seg, axis=0)

        ar = np.arange(L, dtype=np.int32)
        lo_p = np.maximum(0, ar - w)
        cnt_p = (ar - lo_p)[:, None]
        p_seg = np.where(cnt_p > 0, (cs[ar] - cs[lo_p]) / np.maximum(cnt_p, 1), seg)

        lo_f = np.minimum(L, ar + 1)
        hi_f = np.minimum(L, lo_f + w)
        cnt_f = (hi_f - lo_f)[:, None]
        f_seg = np.where(cnt_f > 0, (cs[hi_f] - cs[lo_f]) / np.maximum(cnt_f, 1), seg)

        past[b0:b1] = p_seg.astype(np.float32)
        fut[b0:b1] = f_seg.astype(np.float32)
    return past, fut


def build_dead_seq_features(
    p_dead_s1: np.ndarray,
    Xd_base: np.ndarray,
    blocks: List[Tuple[int, int]],
) -> np.ndarray:
    """Temporal sequence features around Stage-1 dead probability + ball-free kinematics.

    Xd_base[:, :10] contains [mean_spd, std_spd, p10, p25, p50, p75, p90, top3, top2, max]
    plus f_lt15 (col 15), mean_acc (col 20), coh_all (col 25). Strictly ball-free!
    """
    logit_d = _safe_logit(p_dead_s1)
    # Core 1D temporal signals: p_dead, logit_d, mean_spd, p75_spd, top2_spd, f_lt15, coh_all
    sig = np.column_stack([
        p_dead_s1,
        logit_d,
        np.nan_to_num(Xd_base[:, 0], nan=0.0),   # mean_spd
        np.nan_to_num(Xd_base[:, 5], nan=0.0),   # p75_spd
        np.nan_to_num(Xd_base[:, 8], nan=0.0),   # top2_spd
        np.nan_to_num(Xd_base[:, 15], nan=0.0),  # f_lt15
        np.nan_to_num(Xd_base[:, 25], nan=0.0),  # coh_all
    ]).astype(np.float32)

    cols = [sig]
    # Symmetric multi-scale context windows from 0.4s to 21s
    for half in (5, 12, 25, 60, 140, 260):
        cols.append(_roll_mean_blocks(sig[:, :5], blocks, half))
    # Asymmetric one-sided past/future windows and boundary step (future - past)
    for w in (10, 25, 60, 140, 280):
        pm, fm = _one_sided_mean_block(sig[:, :5], blocks, w)
        cols.append(pm)
        cols.append(fm)
        cols.append(fm - pm)
    # Forward-minus-backward boundary derivatives on p_dead and speed
    for w in (10, 25, 60):
        cols.append(_fwd_bwd_diff_blocks(sig[:, :4], blocks, w))
    # Causal vs anticausal EMAs on p_dead_s1 and logit_d
    for alpha in (0.15, 0.05, 0.015):
        cols.append(_causal_ema_blocks(sig[:, :2], blocks, alpha))
        cols.append(_anticausal_ema_blocks(sig[:, :2], blocks, alpha))

    # Ball-free state transition timers (dead-like vs clearly-live frames)
    is_dead = p_dead_s1 > 0.5
    is_live = p_dead_s1 < 0.25
    cols.append(np.column_stack([
        _since_block(is_dead, blocks, cap=400),
        _until_block(is_dead, blocks, cap=400),
        _since_block(is_live, blocks, cap=400),
        _until_block(is_live, blocks, cap=400),
    ]))

    # Local uncertainty / flip rate of Stage-1 dead prob over +/-25f
    dp = np.abs(_fwd_bwd_diff_blocks(p_dead_s1[:, None], blocks, 8))
    cols.append(_roll_mean_blocks(dp, blocks, 25))
    return np.hstack(cols).astype(np.float32)


def build_team_seq_features(
    p_ha_live_s1: np.ndarray,
    p_dead_s1: np.ndarray,
    Xf_base: np.ndarray,
    blocks: List[Tuple[int, int]],
    swap: bool = False,
    Xs_base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """H<->A equivariant Stage-2 temporal sequence features for live team classification.

    Xf_base = Xt_f[:, :65] from features_v4.build_team_features(..., swap=False):
      0: d1_h, 3: d1_a, 15: vor3_h, 32: tti_diff,
      58: bconf, 59: vis_f, 62: bspd, 63: bacc, 64: d_min.
    When Xs_base = Xt_s[:, :65] is provided for swap=True, all team-swapped columns
    match build_team_features(swapped_match, swap=False) bit-for-bit.
    """
    d_ha = np.clip(2.0 * p_ha_live_s1.astype(np.float64) - 1.0, -0.9998, 0.9998)
    sign = -0.5 if swap else 0.5
    p_own = np.clip(0.5 + sign * d_ha, 1e-4, 1.0 - 1e-4).astype(np.float32)
    p_opp = np.clip(0.5 - sign * d_ha, 1e-4, 1.0 - 1e-4).astype(np.float32)
    logit_own = _safe_logit(p_own)
    entropy = -(
        p_own.astype(np.float64) * np.log(p_own.astype(np.float64))
        + p_opp.astype(np.float64) * np.log(p_opp.astype(np.float64))
    ).astype(np.float32)

    if swap and Xs_base is not None:
        src = Xs_base
        d1_own = src[:, 0]
        d1_opp = src[:, 3]
        vor3_own = src[:, 15]
        tti_diff_own = src[:, 32]
    else:
        src = Xf_base
        d1_own = src[:, 3] if swap else src[:, 0]
        d1_opp = src[:, 0] if swap else src[:, 3]
        vor3_own = (1.0 - src[:, 15]) if swap else src[:, 15]
        tti_diff_own = (-src[:, 32]) if swap else src[:, 32]
    bconf = np.nan_to_num(src[:, 58], nan=0.0)
    vis_f = np.nan_to_num(src[:, 59], nan=0.0)
    bspd = np.nan_to_num(src[:, 62], nan=0.0)
    bacc = np.nan_to_num(src[:, 63], nan=0.0)
    d_min = np.nan_to_num(src[:, 64], nan=30.0)

    d1_own_c = np.nan_to_num(d1_own, nan=30.0)
    d1_opp_c = np.nan_to_num(d1_opp, nan=30.0)
    vor3_c = np.nan_to_num(vor3_own, nan=0.5)
    tti_c = np.nan_to_num(tti_diff_own, nan=0.0)

    # First 5 cols drive asymmetric past/future windows; first 4 drive centered rolling means
    sig = np.column_stack([
        p_own,
        logit_own,
        p_dead_s1,
        vor3_c,
        np.tanh(tti_c / 1.5),
        entropy,
        bconf,
        vis_f,
        np.tanh(bspd / 12.0),
        np.tanh(bacc / 15.0),
        np.tanh(d1_own_c / 8.0),
        np.tanh(d1_opp_c / 8.0),
        np.tanh((d1_opp_c - d1_own_c) / 6.0),
        np.tanh(d_min / 8.0),
    ]).astype(np.float32)

    cols = [sig]
    # Symmetric multi-scale context windows
    for half in (5, 12, 25, 55, 120):
        cols.append(_roll_mean_blocks(sig[:, :4], blocks, half))
    # One-sided past / future windows & step contrast (fut - past)
    for w in (8, 18, 40, 85, 160):
        pm, fm = _one_sided_mean_block(sig[:, :5], blocks, w)
        cols.append(pm)
        cols.append(fm)
        cols.append(fm - pm)
    for w in (8, 20, 45):
        cols.append(_fwd_bwd_diff_blocks(sig[:, :2], blocks, w))
    for alpha in (0.20, 0.06, 0.02):
        cols.append(_causal_ema_blocks(sig[:, :2], blocks, alpha))
        cols.append(_anticausal_ema_blocks(sig[:, :2], blocks, alpha))

    # 3-level causal & anticausal contact timers (H<->A swap-equivariant)
    vis_live = vis_f > 0.5
    levels = [
        (vis_live & (d1_own_c < 1.4), vis_live & (d1_opp_c < 1.4)),
        ((bconf > 0.35) & (d1_own_c < 1.8), (bconf > 0.35) & (d1_opp_c < 1.8)),
        ((d1_own_c + 0.5 < d1_opp_c) & (d1_own_c < 2.5), (d1_opp_c + 0.5 < d1_own_c) & (d1_opp_c < 2.5)),
    ]
    timer_cols = []
    for t_own, t_opp in levels:
        so = _since_block(t_own, blocks, cap=300)
        sx = _since_block(t_opp, blocks, cap=300)
        uo = _until_block(t_own, blocks, cap=300)
        ux = _until_block(t_opp, blocks, cap=300)
        t_any = t_own | t_opp
        own_val = np.where(t_own & ~t_opp, 1.0, np.where(t_opp & ~t_own, 0.0, 0.5)).astype(np.float32)
        last_own = _last_val_block(own_val, t_any, blocks, default=0.5)
        next_own = _next_val_block(own_val, t_any, blocks, default=0.5)
        timer_cols.extend([so, sx, uo, ux, sx - so, ux - uo, last_own, next_own])

    # Visibility gap bridges (last visible -> next visible state across burst gaps)
    sv = _since_block(vis_live, blocks, cap=300)
    uv = _until_block(vis_live, blocks, cap=300)
    gap_dur = np.where(vis_live, 0.0, sv + uv).astype(np.float32)
    last_vp = _last_val_block(p_own, vis_live, blocks, default=0.5)
    next_vp = _next_val_block(p_own, vis_live, blocks, default=0.5)
    alpha_gap = np.where(vis_live, 0.5, uv / np.maximum(sv + uv, 1e-4)).astype(np.float32)
    bridge_vp = np.clip(alpha_gap * last_vp + (1.0 - alpha_gap) * next_vp, 1e-4, 1.0 - 1e-4).astype(np.float32)
    last_vor = _last_val_block(vor3_c, vis_live, blocks, default=0.5)
    next_vor = _next_val_block(vor3_c, vis_live, blocks, default=0.5)

    timer_cols.extend([
        sv, uv, gap_dur,
        last_vp, next_vp, bridge_vp,
        _safe_logit(last_vp), _safe_logit(next_vp), _safe_logit(bridge_vp),
        last_vor, next_vor,
    ])
    cols.append(np.column_stack(timer_cols).astype(np.float32))
    return np.hstack(cols).astype(np.float32)


@dataclass
class TemporalStacker:
    dead_s2: object
    team_s2: object

    def predict_log_proba(
        self,
        p_dead_s1: np.ndarray,
        p_ha_live_s1: np.ndarray,
        Xd_base: np.ndarray,
        Xf_base: np.ndarray,
        blocks: List[Tuple[int, int]],
        Xs_base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        Zd = build_dead_seq_features(p_dead_s1, Xd_base, blocks)
        pd = np.clip(self.dead_s2.predict_proba(Zd)[:, 1].astype(np.float64), 1e-4, 1.0 - 1e-4)

        Zt_f = build_team_seq_features(p_ha_live_s1, p_dead_s1, Xf_base, blocks, swap=False)
        Zt_s = build_team_seq_features(p_ha_live_s1, p_dead_s1, Xf_base, blocks, swap=True, Xs_base=Xs_base)
        ph = self.team_s2.predict_proba(Zt_f)[:, 1].astype(np.float64)
        pa = self.team_s2.predict_proba(Zt_s)[:, 1].astype(np.float64)
        d_s2 = (ph - pa) / np.maximum(ph + pa, 1e-6)
        ph_live = np.clip(0.5 + 0.5 * d_s2, 1e-4, 1.0 - 1e-4)
        pa_live = np.clip(0.5 - 0.5 * d_s2, 1e-4, 1.0 - 1e-4)

        live = 1.0 - pd
        p3 = np.column_stack([live * ph_live, live * pa_live, pd])
        return np.log(np.clip(p3, 1e-5, 1.0)).astype(np.float32)
