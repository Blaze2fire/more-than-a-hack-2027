"""Fast vectorized feature extraction for Stage-1 dead-ball and team-possession models.

Design principles:
- Zero ball leakage in build_dead_features(): dead-ball evidence uses ONLY player
  positions, velocities, accelerations, collective coherence, formation geometry,
  and asymmetric forward/backward transition change-points.
- Full ball-reconstruction integration in build_team_features(): consumes
  ReconstructedBall (ball_hat, ball_conf, ball_vel, ball_acc, gap_len) with
  closing speeds, time-to-intercept, forward trajectory projections, Voronoi pitch
  control, and causal/anticausal touch ownership.
- Equivariant under H <-> A swap: build_team_features(..., swap=True) swaps own vs opp
  so training on (H, swap=False) + (A, swap=True) yields strictly team-symmetric scores.
"""
from typing import List, Optional, Tuple

import numpy as np

from solution.ball_tracker import ReconstructedBall, compute_player_vel, reconstruct_ball
from solution.common import FPS, Match

CHUNK_FRAMES = 4096


def _roll_mean_blocks(X: np.ndarray, blocks: List[Tuple[int, int]], half: int) -> np.ndarray:
    """Exact prefix-sum centered rolling mean [t-half, t+half] within contiguous blocks."""
    n, f = X.shape
    out = np.zeros_like(X, dtype=np.float32)
    X64 = np.nan_to_num(X, nan=0.0).astype(np.float64, copy=False)
    for b0, b1 in blocks:
        nb = b1 - b0
        if nb <= 0:
            continue
        cs = np.zeros((nb + 1, f), dtype=np.float64)
        np.cumsum(X64[b0:b1], axis=0, out=cs[1:])
        idx = np.arange(nb)
        lo = np.maximum(0, idx - half)
        hi = np.minimum(nb, idx + half + 1)
        out[b0:b1] = ((cs[hi] - cs[lo]) / (hi - lo)[:, None]).astype(np.float32)
    return out


def _fwd_bwd_diff_blocks(X: np.ndarray, blocks: List[Tuple[int, int]], w: int) -> np.ndarray:
    """Forward mean [t, t+w) minus backward mean [t-w, t) per block (boundary detector)."""
    n, f = X.shape
    out = np.zeros_like(X, dtype=np.float32)
    X64 = np.nan_to_num(X, nan=0.0).astype(np.float64, copy=False)
    for b0, b1 in blocks:
        nb = b1 - b0
        if nb <= 1:
            continue
        cs = np.zeros((nb + 1, f), dtype=np.float64)
        np.cumsum(X64[b0:b1], axis=0, out=cs[1:])
        idx = np.arange(nb)
        b_lo = np.maximum(0, idx - w)
        b_cnt = np.maximum(1, idx - b_lo)[:, None]
        b_mean = (cs[idx] - cs[b_lo]) / b_cnt

        f_hi = np.minimum(nb, idx + w)
        f_cnt = np.maximum(1, f_hi - idx)[:, None]
        f_mean = (cs[f_hi] - cs[idx]) / f_cnt

        diff = f_mean - b_mean
        diff[idx == 0] = 0.0
        out[b0:b1] = diff.astype(np.float32)
    return out


def build_dead_features(
    m: Match,
    vel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract strictly ball-free player-kinematic & formation features for dead-ball detection."""
    n, s, _ = m.pos.shape
    pitch_l, pitch_w = m.pitch
    if vel is None:
        vel = compute_player_vel(m.pos, m.blocks)

    px = m.pos[:, :, 0].astype(np.float32, copy=False)
    py = m.pos[:, :, 1].astype(np.float32, copy=False)
    vx = np.nan_to_num(vel[:, :, 0], nan=0.0).astype(np.float32, copy=False)
    vy = np.nan_to_num(vel[:, :, 1], nan=0.0).astype(np.float32, copy=False)
    spd = m.spd.astype(np.float32, copy=False)

    # Player vector acceleration magnitude (|dv/dt| over +/-3 frames = 0.24s)
    acc = np.zeros((n, s), dtype=np.float32)
    for b0, b1 in m.blocks:
        if b1 - b0 > 6:
            dvx = (vx[b0 + 6 : b1] - vx[b0 : b1 - 6]) * (FPS / 6.0)
            dvy = (vy[b0 + 6 : b1] - vy[b0 : b1 - 6]) * (FPS / 6.0)
            acc[b0 + 3 : b1 - 3] = np.clip(np.hypot(dvx, dvy), 0.0, 25.0)

    valid_p = np.isfinite(px) & np.isfinite(py)
    valid_s = valid_p & np.isfinite(spd)
    th = m.team == 0
    ta = m.team == 1
    same_team = (m.team[:, None] == m.team[None, :])

    # Pre-allocate instantaneous dead features (34 base scalars)
    mean_spd = np.zeros(n, dtype=np.float32)
    std_spd = np.zeros(n, dtype=np.float32)
    p10_spd = np.zeros(n, dtype=np.float32)
    p25_spd = np.zeros(n, dtype=np.float32)
    p50_spd = np.zeros(n, dtype=np.float32)
    p75_spd = np.zeros(n, dtype=np.float32)
    p90_spd = np.zeros(n, dtype=np.float32)
    top3_spd = np.zeros(n, dtype=np.float32)
    top2_spd = np.zeros(n, dtype=np.float32)
    max_spd = np.zeros(n, dtype=np.float32)

    min_team_mean_spd = np.zeros(n, dtype=np.float32)
    max_team_mean_spd = np.zeros(n, dtype=np.float32)
    diff_team_mean_spd = np.zeros(n, dtype=np.float32)
    max_team_top_spd = np.zeros(n, dtype=np.float32)

    f_lt08 = np.zeros(n, dtype=np.float32)
    f_lt15 = np.zeros(n, dtype=np.float32)
    f_mid = np.zeros(n, dtype=np.float32)
    f_ge30 = np.zeros(n, dtype=np.float32)
    f_ge45 = np.zeros(n, dtype=np.float32)
    f_ge60 = np.zeros(n, dtype=np.float32)

    mean_acc = np.zeros(n, dtype=np.float32)
    p75_acc = np.zeros(n, dtype=np.float32)
    max_acc = np.zeros(n, dtype=np.float32)
    f_acc_hi = np.zeros(n, dtype=np.float32)

    coll_spd = np.zeros(n, dtype=np.float32)
    coh_all = np.zeros(n, dtype=np.float32)
    max_team_coh = np.zeros(n, dtype=np.float32)

    std_x = np.full(n, 14.0, dtype=np.float32)
    std_y = np.full(n, 12.0, dtype=np.float32)
    spread_x = np.full(n, 35.0, dtype=np.float32)
    spread_y = np.full(n, 30.0, dtype=np.float32)
    team_sep_x = np.zeros(n, dtype=np.float32)
    team_comp_min = np.full(n, 12.0, dtype=np.float32)
    team_comp_max = np.full(n, 18.0, dtype=np.float32)
    min_opp_d = np.full(n, 5.0, dtype=np.float32)
    frac_marked_3m = np.zeros(n, dtype=np.float32)
    frac_box = np.zeros(n, dtype=np.float32)
    frac_sideline = np.zeros(n, dtype=np.float32)
    dens_edge_y = np.full(n, 20.0, dtype=np.float32)
    dens_edge_x = np.full(n, 35.0, dtype=np.float32)

    for start in range(0, n, CHUNK_FRAMES):
        end = min(n, start + CHUNK_FRAMES)
        m_len = end - start
        vp = valid_p[start:end]
        vs = valid_s[start:end]
        px_c = px[start:end]
        py_c = py[start:end]
        sp_c = spd[start:end]
        vx_c = vx[start:end]
        vy_c = vy[start:end]
        ac_c = acc[start:end]

        cnt_s = np.maximum(vs.sum(axis=1), 1)
        sp_z = np.where(vs, sp_c, 0.0)
        ms = sp_z.sum(axis=1) / cnt_s
        mean_spd[start:end] = ms
        var_s = np.where(vs, (sp_z - ms[:, None]) ** 2, 0.0).sum(axis=1) / cnt_s
        std_spd[start:end] = np.sqrt(var_s)

        # Sort valid speeds ascending (NaN goes to end)
        sp_nan = np.where(vs, sp_c, np.nan)
        sp_sorted = np.sort(sp_nan, axis=1)
        row_ar = np.arange(m_len)
        max_idx = np.clip(vs.sum(axis=1) - 1, 0, s - 1)
        def _q(frac: float) -> np.ndarray:
            qi = np.clip(np.rint(frac * max_idx).astype(np.int32), 0, s - 1)
            return np.nan_to_num(sp_sorted[row_ar, qi], nan=0.0)
        p10_spd[start:end] = _q(0.10)
        p25_spd[start:end] = _q(0.25)
        p50_spd[start:end] = _q(0.50)
        p75_spd[start:end] = _q(0.75)
        p90_spd[start:end] = _q(0.90)
        top3_spd[start:end] = np.nan_to_num(sp_sorted[row_ar, np.maximum(0, max_idx - 2)], nan=0.0)
        top2_spd[start:end] = np.nan_to_num(sp_sorted[row_ar, np.maximum(0, max_idx - 1)], nan=0.0)
        max_spd[start:end] = np.nan_to_num(sp_sorted[row_ar, max_idx], nan=0.0)

        f_lt08[start:end] = (vs & (sp_c < 0.8)).sum(axis=1) / cnt_s
        f_lt15[start:end] = (vs & (sp_c < 1.5)).sum(axis=1) / cnt_s
        f_mid[start:end] = (vs & (sp_c >= 1.5) & (sp_c < 3.0)).sum(axis=1) / cnt_s
        f_ge30[start:end] = (vs & (sp_c >= 3.0)).sum(axis=1) / cnt_s
        f_ge45[start:end] = (vs & (sp_c >= 4.5)).sum(axis=1) / cnt_s
        f_ge60[start:end] = (vs & (sp_c >= 6.0)).sum(axis=1) / cnt_s

        # Team-level speeds
        vsh = vs & th[None, :]
        vsa = vs & ta[None, :]
        mh = np.where(vsh, sp_z, 0.0).sum(axis=1) / np.maximum(vsh.sum(axis=1), 1)
        ma = np.where(vsa, sp_z, 0.0).sum(axis=1) / np.maximum(vsa.sum(axis=1), 1)
        min_team_mean_spd[start:end] = np.minimum(mh, ma)
        max_team_mean_spd[start:end] = np.maximum(mh, ma)
        diff_team_mean_spd[start:end] = np.abs(mh - ma)
        top_h = np.where(vsh, sp_z, 0.0).max(axis=1)
        top_a = np.where(vsa, sp_z, 0.0).max(axis=1)
        max_team_top_spd[start:end] = np.maximum(top_h, top_a)

        # Acceleration features
        ac_nan = np.where(vs, ac_c, np.nan)
        ac_sorted = np.sort(ac_nan, axis=1)
        ac_z = np.where(vs, ac_c, 0.0)
        mean_acc[start:end] = ac_z.sum(axis=1) / cnt_s
        p75_acc[start:end] = np.nan_to_num(ac_sorted[row_ar, np.clip(np.rint(0.75 * max_idx).astype(np.int32), 0, s - 1)], nan=0.0)
        max_acc[start:end] = np.nan_to_num(ac_sorted[row_ar, max_idx], nan=0.0)
        f_acc_hi[start:end] = (vs & (ac_c > 2.5)).sum(axis=1) / cnt_s

        # Collective velocity coherence
        vx_mean = np.where(vs, vx_c, 0.0).sum(axis=1) / cnt_s
        vy_mean = np.where(vs, vy_c, 0.0).sum(axis=1) / cnt_s
        cm = np.hypot(vx_mean, vy_mean)
        coll_spd[start:end] = cm
        coh_all[start:end] = cm / (ms + 0.3)

        vh_vx = np.where(vsh, vx_c, 0.0).sum(axis=1) / np.maximum(vsh.sum(axis=1), 1)
        vh_vy = np.where(vsh, vy_c, 0.0).sum(axis=1) / np.maximum(vsh.sum(axis=1), 1)
        va_vx = np.where(vsa, vx_c, 0.0).sum(axis=1) / np.maximum(vsa.sum(axis=1), 1)
        va_vy = np.where(vsa, vy_c, 0.0).sum(axis=1) / np.maximum(vsa.sum(axis=1), 1)
        coh_h = np.hypot(vh_vx, vh_vy) / (mh + 0.3)
        coh_a = np.hypot(va_vx, va_vy) / (ma + 0.3)
        max_team_coh[start:end] = np.maximum(coh_h, coh_a)

        # Formation geometry & pitch zones
        cnt_p = np.maximum(vp.sum(axis=1), 1)
        px0 = np.where(vp, px_c, 0.0)
        py0 = np.where(vp, py_c, 0.0)
        px_sort = np.sort(np.where(vp, px_c, np.nan), axis=1)
        py_sort = np.sort(np.where(vp, py_c, np.nan), axis=1)
        p_max_idx = np.clip(vp.sum(axis=1) - 1, 0, s - 1)
        q10_i = np.clip(np.rint(0.10 * p_max_idx).astype(np.int32), 0, s - 1)
        q90_i = np.clip(np.rint(0.90 * p_max_idx).astype(np.int32), 0, s - 1)
        spread_x[start:end] = np.nan_to_num(px_sort[row_ar, q90_i] - px_sort[row_ar, q10_i], nan=35.0)
        spread_y[start:end] = np.nan_to_num(py_sort[row_ar, q90_i] - py_sort[row_ar, q10_i], nan=30.0)

        mux = px0.sum(axis=1) / cnt_p
        muy = py0.sum(axis=1) / cnt_p
        std_x[start:end] = np.sqrt(np.where(vp, (px0 - mux[:, None]) ** 2, 0.0).sum(axis=1) / cnt_p)
        std_y[start:end] = np.sqrt(np.where(vp, (py0 - muy[:, None]) ** 2, 0.0).sum(axis=1) / cnt_p)

        vph = vp & th[None, :]
        vpa = vp & ta[None, :]
        ch_cnt = np.maximum(vph.sum(axis=1), 1)
        ca_cnt = np.maximum(vpa.sum(axis=1), 1)
        hx = np.where(vph, px0, 0.0).sum(axis=1) / ch_cnt
        hy = np.where(vph, py0, 0.0).sum(axis=1) / ch_cnt
        ax = np.where(vpa, px0, 0.0).sum(axis=1) / ca_cnt
        ay = np.where(vpa, py0, 0.0).sum(axis=1) / ca_cnt
        team_sep_x[start:end] = np.abs(hx - ax)
        comp_h = np.sqrt(np.where(vph, (px0 - hx[:, None]) ** 2 + (py0 - hy[:, None]) ** 2, 0.0).sum(axis=1) / ch_cnt)
        comp_a = np.sqrt(np.where(vpa, (px0 - ax[:, None]) ** 2 + (py0 - ay[:, None]) ** 2, 0.0).sum(axis=1) / ca_cnt)
        team_comp_min[start:end] = np.minimum(comp_h, comp_a)
        team_comp_max[start:end] = np.maximum(comp_h, comp_a)

        in_box = vp & ((px0 < 18.0) | (px0 > pitch_l - 18.0)) & (np.abs(py0 - 0.5 * pitch_w) < 22.0)
        frac_box[start:end] = in_box.sum(axis=1) / cnt_p
        near_side = vp & (np.minimum(py0, pitch_w - py0) < 6.0)
        frac_sideline[start:end] = near_side.sum(axis=1) / cnt_p

        # Pairwise opposing proximity & density center
        both_v = vp[:, :, None] & vp[:, None, :]
        d2 = (px0[:, :, None] - px0[:, None, :]) ** 2 + (py0[:, :, None] - py0[:, None, :]) ** 2
        d2_opp = np.where(both_v & (~same_team[None, :, :]), d2, np.float32(1e6))
        min_opp = np.sqrt(np.clip(d2_opp.min(axis=2), 0.0, 1e6))  # (B, S)
        min_opp_valid = np.where(vp, min_opp, 1e3)
        min_opp_d[start:end] = np.clip(min_opp_valid.min(axis=1), 0.0, 50.0)
        frac_marked_3m[start:end] = (vp & (min_opp < 3.0)).sum(axis=1) / cnt_p

        k8 = np.where(both_v, np.exp(-d2 / np.float32(128.0)), np.float32(0.0)).sum(axis=2)
        w8 = np.where(vp, k8 ** 2 + 1e-4, 0.0)
        w8s = np.maximum(w8.sum(axis=1), 1e-6)
        dx_pk = (w8 * px0).sum(axis=1) / w8s
        dy_pk = (w8 * py0).sum(axis=1) / w8s
        dens_edge_x[start:end] = np.minimum(dx_pk, pitch_l - dx_pk)
        dens_edge_y[start:end] = np.minimum(dy_pk, pitch_w - dy_pk)

    base = np.column_stack([
        mean_spd, std_spd, p10_spd, p25_spd, p50_spd, p75_spd, p90_spd,
        top3_spd, top2_spd, max_spd,
        min_team_mean_spd, max_team_mean_spd, diff_team_mean_spd, max_team_top_spd,
        f_lt08, f_lt15, f_mid, f_ge30, f_ge45, f_ge60,
        mean_acc, p75_acc, max_acc, f_acc_hi,
        coll_spd, coh_all, max_team_coh,
        std_x, std_y, spread_x, spread_y, team_sep_x, team_comp_min, team_comp_max,
        min_opp_d, frac_marked_3m, frac_box, frac_sideline, dens_edge_x, dens_edge_y,
    ]).astype(np.float32)

    # Core dynamic indicators summarized across multiple temporal windows
    dyn_cols = np.column_stack([
        mean_spd, p25_spd, p50_spd, p75_spd, top2_spd, max_spd,
        f_lt08, f_lt15, f_ge30, f_ge45,
        mean_acc, f_acc_hi, coll_spd, coh_all,
        std_x, std_y, min_opp_d, frac_marked_3m,
    ]).astype(np.float32)

    cols = [base]
    # Centered macro-to-micro temporal context
    m15 = _roll_mean_blocks(dyn_cols, m.blocks, 15)
    m150 = _roll_mean_blocks(dyn_cols, m.blocks, 150)
    for half in (6, 15, 35, 75, 150, 300):
        cols.append(_roll_mean_blocks(dyn_cols, m.blocks, half))
    # Short-vs-long contrast (momentary slow-up in live play vs real stoppage)
    cols.append(m15 - m150)

    # Boundary change-point detectors: forward - backward on key speed/acc/coherence cues
    chg_cols = np.column_stack([
        mean_spd, p50_spd, top2_spd, f_lt15, f_ge30, f_ge45, mean_acc, coll_spd, coh_all,
    ]).astype(np.float32)
    for w in (12, 30, 75):
        cols.append(_fwd_bwd_diff_blocks(chg_cols, m.blocks, w))

    return np.hstack(cols).astype(np.float32)


def _last_next_touch(
    margin: np.ndarray,
    touch_mask: np.ndarray,
    blocks: List[Tuple[int, int]],
    cap: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal last-touch and anticausal next-touch signed control margin & frame distances."""
    n = len(margin)
    last_val = np.zeros(n, dtype=np.float32)
    next_val = np.zeros(n, dtype=np.float32)
    t_since = np.full(n, cap, dtype=np.float32)
    t_until = np.full(n, cap, dtype=np.float32)

    for b0, b1 in blocks:
        cur_v = 0.0
        cur_age = cap
        for t in range(b0, b1):
            if touch_mask[t]:
                cur_v = float(margin[t])
                cur_age = 0.0
            else:
                cur_age = min(cap, cur_age + 1.0)
            last_val[t] = cur_v
            t_since[t] = cur_age

        nxt_v = 0.0
        nxt_age = cap
        for t in range(b1 - 1, b0 - 1, -1):
            if touch_mask[t]:
                nxt_v = float(margin[t])
                nxt_age = 0.0
            else:
                nxt_age = min(cap, nxt_age + 1.0)
            next_val[t] = nxt_v
            t_until[t] = nxt_age

    return last_val, t_since, next_val, t_until


def build_team_features(
    m: Match,
    rb: Optional[ReconstructedBall] = None,
    vel: Optional[np.ndarray] = None,
    swap: bool = False,
    use_p3: bool = False,
) -> np.ndarray:
    """Build equivariant own-vs-opp features for live team possession estimation."""
    n, s, _ = m.pos.shape
    pitch_l, pitch_w = m.pitch
    if vel is None:
        vel = compute_player_vel(m.pos, m.blocks)
    if rb is None:
        rb = reconstruct_ball(m, imputer=None, vel=vel)

    own_team = 1 if swap else 0
    opp_team = 0 if swap else 1
    mask_own = (m.team == own_team)
    mask_opp = (m.team == opp_team)

    px = m.pos[:, :, 0].astype(np.float32, copy=False)
    py = m.pos[:, :, 1].astype(np.float32, copy=False)
    vx = np.nan_to_num(vel[:, :, 0], nan=0.0).astype(np.float32, copy=False)
    vy = np.nan_to_num(vel[:, :, 1], nan=0.0).astype(np.float32, copy=False)
    vp = np.isfinite(px) & np.isfinite(py)
    vp_own = vp & mask_own[None, :]
    vp_opp = vp & mask_opp[None, :]

    bx = rb.ball_hat[:, 0].astype(np.float32, copy=False)
    by = rb.ball_hat[:, 1].astype(np.float32, copy=False)
    bvx = rb.ball_vel[:, 0].astype(np.float32, copy=False)
    bvy = rb.ball_vel[:, 1].astype(np.float32, copy=False)
    bconf = rb.ball_conf.astype(np.float32, copy=False)
    bspd = rb.ball_spd.astype(np.float32, copy=False)
    bacc = rb.ball_acc.astype(np.float32, copy=False)

    # Pre-allocate per-frame outputs
    d1_own = np.full(n, 40.0, dtype=np.float32)
    d2_own = np.full(n, 50.0, dtype=np.float32)
    d3_own = np.full(n, 60.0, dtype=np.float32)
    d1_opp = np.full(n, 40.0, dtype=np.float32)
    d2_opp = np.full(n, 50.0, dtype=np.float32)
    d3_opp = np.full(n, 60.0, dtype=np.float32)

    vor_own = {T: np.full(n, 0.5, dtype=np.float32) for T in (1.5, 3.0, 6.0, 12.0)}
    n2_own = np.zeros(n, dtype=np.float32)
    n2_opp = np.zeros(n, dtype=np.float32)
    n5_own = np.zeros(n, dtype=np.float32)
    n5_opp = np.zeros(n, dtype=np.float32)
    n10_own = np.zeros(n, dtype=np.float32)
    n10_opp = np.zeros(n, dtype=np.float32)

    close1_own = np.zeros(n, dtype=np.float32)
    close1_opp = np.zeros(n, dtype=np.float32)
    tti1_own = np.full(n, 15.0, dtype=np.float32)
    tti1_opp = np.full(n, 15.0, dtype=np.float32)
    tti2_own = np.full(n, 18.0, dtype=np.float32)
    tti2_opp = np.full(n, 18.0, dtype=np.float32)

    cos1_own = np.zeros(n, dtype=np.float32)
    cos1_opp = np.zeros(n, dtype=np.float32)
    fut_d_own = {tau: np.full(n, 40.0, dtype=np.float32) for tau in (0.25, 0.50, 1.00)}
    fut_d_opp = {tau: np.full(n, 40.0, dtype=np.float32) for tau in (0.25, 0.50, 1.00)}
    vor_fut05 = np.full(n, 0.5, dtype=np.float32)

    cent_dist_own = np.full(n, 30.0, dtype=np.float32)
    cent_dist_opp = np.full(n, 30.0, dtype=np.float32)

    for start in range(0, n, CHUNK_FRAMES):
        end = min(n, start + CHUNK_FRAMES)
        m_len = end - start
        row_ar = np.arange(m_len)

        dx_b = px[start:end] - bx[start:end, None]
        dy_b = py[start:end] - by[start:end, None]
        dist = np.hypot(dx_b, dy_b)

        vo = vp_own[start:end]
        va = vp_opp[start:end]
        dist_own = np.where(vo, dist, np.float32(1e4))
        dist_opp = np.where(va, dist, np.float32(1e4))

        s_own = np.sort(dist_own, axis=1)
        s_opp = np.sort(dist_opp, axis=1)
        d1_own[start:end] = np.minimum(s_own[:, 0], 80.0)
        d2_own[start:end] = np.minimum(s_own[:, min(1, s - 1)], 85.0)
        d3_own[start:end] = np.minimum(s_own[:, min(2, s - 1)], 90.0)
        d1_opp[start:end] = np.minimum(s_opp[:, 0], 80.0)
        d2_opp[start:end] = np.minimum(s_opp[:, min(1, s - 1)], 85.0)
        d3_opp[start:end] = np.minimum(s_opp[:, min(2, s - 1)], 90.0)

        n2_own[start:end] = (dist_own < 2.0).sum(axis=1)
        n2_opp[start:end] = (dist_opp < 2.0).sum(axis=1)
        n5_own[start:end] = (dist_own < 5.0).sum(axis=1)
        n5_opp[start:end] = (dist_opp < 5.0).sum(axis=1)
        n10_own[start:end] = (dist_own < 10.0).sum(axis=1)
        n10_opp[start:end] = (dist_opp < 10.0).sum(axis=1)

        # Shift by frame minimum so exp never underflows to 0/0 when ball is far
        d_min_both = np.minimum(s_own[:, 0], s_opp[:, 0])
        for T in (1.5, 3.0, 6.0, 12.0):
            pot_o = np.where(vo, np.exp(-(dist - d_min_both[:, None]) / np.float32(T)), np.float32(0.0)).sum(axis=1)
            pot_a = np.where(va, np.exp(-(dist - d_min_both[:, None]) / np.float32(T)), np.float32(0.0)).sum(axis=1)
            vor_own[T][start:end] = pot_o / np.maximum(pot_o + pot_a, 1e-6)

        # Nearest player closing velocity & alignment
        idx1_o = np.argmin(dist_own, axis=1)
        idx1_a = np.argmin(dist_opp, axis=1)
        inv_do = 1.0 / np.maximum(dist_own[row_ar, idx1_o], 0.1)
        inv_da = 1.0 / np.maximum(dist_opp[row_ar, idx1_a], 0.1)
        px_c = np.nan_to_num(px[start:end], nan=0.0)
        py_c = np.nan_to_num(py[start:end], nan=0.0)
        vx_c = vx[start:end]
        vy_c = vy[start:end]
        ux_o = (bx[start:end] - px_c[row_ar, idx1_o]) * inv_do
        uy_o = (by[start:end] - py_c[row_ar, idx1_o]) * inv_do
        ux_a = (bx[start:end] - px_c[row_ar, idx1_a]) * inv_da
        uy_a = (by[start:end] - py_c[row_ar, idx1_a]) * inv_da

        rvx_o = vx_c[row_ar, idx1_o] - bvx[start:end]
        rvy_o = vy_c[row_ar, idx1_o] - bvy[start:end]
        rvx_a = vx_c[row_ar, idx1_a] - bvx[start:end]
        rvy_a = vy_c[row_ar, idx1_a] - bvy[start:end]
        close1_own[start:end] = np.where(vo.any(axis=1), rvx_o * ux_o + rvy_o * uy_o, 0.0)
        close1_opp[start:end] = np.where(va.any(axis=1), rvx_a * ux_a + rvy_a * uy_a, 0.0)

        # Ball direction cosine with player vector (bx - px vs ball_vel) -> -ux means ball travels toward player
        bv_norm = np.maximum(bspd[start:end], 0.5)
        bux = bvx[start:end] / bv_norm
        buy = bvy[start:end] / bv_norm
        cos1_own[start:end] = -(ux_o * bux + uy_o * buy)
        cos1_opp[start:end] = -(ux_a * bux + uy_a * buy)

        # Estimated Time-to-Intercept across all team players
        inv_d_all = 1.0 / np.maximum(dist, 0.2)
        ux_all = -dx_b * inv_d_all
        uy_all = -dy_b * inv_d_all
        v_toward = vx[start:end] * ux_all + vy[start:end] * uy_all
        eff_spd = np.clip(2.2 + 0.8 * v_toward, 0.8, 9.5)
        tti = dist / eff_spd
        tti_o = np.sort(np.where(vo, tti, np.float32(50.0)), axis=1)
        tti_a = np.sort(np.where(va, tti, np.float32(50.0)), axis=1)
        tti1_own[start:end] = np.minimum(tti_o[:, 0], 20.0)
        tti1_opp[start:end] = np.minimum(tti_a[:, 0], 20.0)
        tti2_own[start:end] = np.minimum(tti_o[:, min(1, s - 1)], 25.0)
        tti2_opp[start:end] = np.minimum(tti_a[:, min(1, s - 1)], 25.0)

        # Forward trajectory projections
        conf_scale = 0.25 + 0.75 * bconf[start:end]
        for tau in (0.25, 0.50, 1.00):
            fx = np.clip(bx[start:end] + tau * conf_scale * bvx[start:end], -2.0, pitch_l + 2.0)
            fy = np.clip(by[start:end] + tau * conf_scale * bvy[start:end], -2.0, pitch_w + 2.0)
            df = np.hypot(px[start:end] - fx[:, None], py[start:end] - fy[:, None])
            dfo = np.where(vo, df, np.float32(1e4))
            dfa = np.where(va, df, np.float32(1e4))
            fut_d_own[tau][start:end] = np.minimum(dfo.min(axis=1), 80.0)
            fut_d_opp[tau][start:end] = np.minimum(dfa.min(axis=1), 80.0)
            if tau == 0.50:
                df_min = np.minimum(fut_d_own[tau][start:end], fut_d_opp[tau][start:end])
                po = np.where(vo, np.exp(-(df - df_min[:, None]) / 3.0), 0.0).sum(axis=1)
                pa = np.where(va, np.exp(-(df - df_min[:, None]) / 3.0), 0.0).sum(axis=1)
                vor_fut05[start:end] = po / np.maximum(po + pa, 1e-6)

        # Team centroid distances to ball
        co_cnt = np.maximum(vo.sum(axis=1), 1)
        ca_cnt = np.maximum(va.sum(axis=1), 1)
        cx_o = np.where(vo, px[start:end], 0.0).sum(axis=1) / co_cnt
        cy_o = np.where(vo, py[start:end], 0.0).sum(axis=1) / co_cnt
        cx_a = np.where(va, px[start:end], 0.0).sum(axis=1) / ca_cnt
        cy_a = np.where(va, py[start:end], 0.0).sum(axis=1) / ca_cnt
        cent_dist_own[start:end] = np.hypot(cx_o - bx[start:end], cy_o - by[start:end])
        cent_dist_opp[start:end] = np.hypot(cx_a - bx[start:end], cy_a - by[start:end])

    # Core signed differences & confidence interactions
    d1_diff = d1_opp - d1_own
    d2_diff = d2_opp - d2_own
    d3_diff = d3_opp - d3_own
    tti_diff = tti1_opp - tti1_own
    tti2_diff = tti2_opp - tti2_own
    ctrl_sharp = np.exp(-d1_own / 1.8) - np.exp(-d1_opp / 1.8)
    ctrl_conf = ctrl_sharp * bconf
    d1_conf = np.clip(d1_diff, -12.0, 12.0) * bconf
    spike_ctrl = np.clip(d1_diff, -10.0, 10.0) * np.tanh(bacc / 12.0) * bconf
    drib_ctrl = ctrl_sharp * (bspd < 8.0).astype(np.float32) * bconf

    fut025_diff = fut_d_opp[0.25] - fut_d_own[0.25]
    fut050_diff = fut_d_opp[0.50] - fut_d_own[0.50]
    fut100_diff = fut_d_opp[1.00] - fut_d_own[1.00]

    # Causal and anticausal clear-touch ownership trackers
    d_min = np.minimum(d1_own, d1_opp)
    touch_ok = (bconf > 0.65) & (d_min < 2.6) & ((bspd < 10.0) | (bacc > 10.0))
    touch_margin = np.clip(d1_diff, -8.0, 8.0) * bconf
    last_m, t_since, next_m, t_until = _last_next_touch(touch_margin, touch_ok, m.blocks, cap=250.0)
    # Continuous interpolation from last touch -> next touch (reception semantics: stays with last until next arrives!)
    touch_ratio = t_until / (t_since + t_until + 1.0)
    touch_same = last_m * next_m

    log_gap = np.log1p(rb.gap_len).astype(np.float32)
    log_dvis = np.log1p(rb.dist_to_vis).astype(np.float32)
    vis_f = rb.vis_clean.astype(np.float32)

    base = np.column_stack([
        d1_own, d2_own, d3_own,
        d1_opp, d2_opp, d3_opp,
        d1_diff, d2_diff, d3_diff,
        ctrl_sharp, ctrl_conf, d1_conf, spike_ctrl, drib_ctrl,
        vor_own[1.5], vor_own[3.0], vor_own[6.0], vor_own[12.0],
        n2_own, n2_opp, n2_own - n2_opp,
        n5_own, n5_opp, n5_own - n5_opp,
        n10_own, n10_opp, n10_own - n10_opp,
        close1_own, close1_opp, close1_own - close1_opp,
        tti1_own, tti1_opp, tti_diff,
        tti2_own, tti2_opp, tti2_diff,
        cos1_own, cos1_opp, cos1_own - cos1_opp,
        fut_d_own[0.25], fut_d_opp[0.25], fut025_diff,
        fut_d_own[0.50], fut_d_opp[0.50], fut050_diff, vor_fut05,
        fut_d_own[1.00], fut_d_opp[1.00], fut100_diff,
        cent_dist_own, cent_dist_opp, cent_dist_opp - cent_dist_own,
        last_m, t_since, next_m, t_until, touch_ratio, touch_same,
        bconf, vis_f, log_gap, log_dvis, bspd, bacc, d_min,
    ]).astype(np.float32)

    # Multi-scale temporal smoothing + change-point contrast on core signed possession cues
    temp_sig = np.column_stack([
        ctrl_sharp, ctrl_conf, d1_conf, spike_ctrl, drib_ctrl,
        vor_own[1.5], vor_own[3.0], vor_own[6.0], vor_own[12.0],
        tti_diff, fut050_diff, vor_fut05, last_m, next_m,
    ]).astype(np.float32)

    cols = [base]
    for half in (6, 15, 35, 80, 160):
        cols.append(_roll_mean_blocks(temp_sig, m.blocks, half))

    # Boundary sharpness: forward - backward differences on fast team-control indicators
    chg_sig = np.column_stack([
        ctrl_sharp, ctrl_conf, vor_own[1.5], vor_own[3.0], vor_own[6.0], tti_diff, fut050_diff,
    ]).astype(np.float32)
    for w in (10, 25, 60):
        cols.append(_fwd_bwd_diff_blocks(chg_sig, m.blocks, w))

    if use_p3:
        cols.append(compute_centroid_axis_motion(m, vel=vel, swap=swap))
        cols.append(compute_deblurred_ball_features(m, rb=rb, px=px, py=py, vp_own=vp_own, vp_opp=vp_opp))

    return np.hstack(cols).astype(np.float32)


def compute_centroid_axis_motion(
    m: Match,
    vel: Optional[np.ndarray] = None,
    swap: bool = False,
) -> np.ndarray:
    """Strictly ball-free, H<->A equivariant opposing centroid-axis motion features (45 cols)."""
    n = m.n
    if vel is None:
        vel = compute_player_vel(m.pos, m.blocks)
    own_team = 1 if swap else 0
    opp_team = 0 if swap else 1
    mask_own = (m.team == own_team)
    mask_opp = (m.team == opp_team)

    px = m.pos[:, :, 0].astype(np.float32, copy=False)
    py = m.pos[:, :, 1].astype(np.float32, copy=False)
    vx = np.nan_to_num(vel[:, :, 0], nan=0.0).astype(np.float32, copy=False)
    vy = np.nan_to_num(vel[:, :, 1], nan=0.0).astype(np.float32, copy=False)
    spd_p = np.hypot(vx, vy)
    vp = np.isfinite(px) & np.isfinite(py)
    vp_own = vp & mask_own[None, :]
    vp_opp = vp & mask_opp[None, :]

    def _team_stats(v_mask: np.ndarray):
        cnt = np.maximum(v_mask.sum(axis=1).astype(np.float32), 1.0)
        has = v_mask.any(axis=1)
        mx = np.where(v_mask, px, 0.0).sum(axis=1) / cnt
        my = np.where(v_mask, py, 0.0).sum(axis=1) / cnt
        cx = np.where(has, mx, 0.5 * m.pitch[0]).astype(np.float32)
        cy = np.where(has, my, 0.5 * m.pitch[1]).astype(np.float32)
        mvx = (np.where(v_mask, vx, 0.0).sum(axis=1) / cnt).astype(np.float32)
        mvy = (np.where(v_mask, vy, 0.0).sum(axis=1) / cnt).astype(np.float32)
        mspd = (np.where(v_mask, spd_p, 0.0).sum(axis=1) / cnt).astype(np.float32)
        dx = np.where(v_mask, px - cx[:, None], 0.0)
        dy = np.where(v_mask, py - cy[:, None], 0.0)
        return cx, cy, mvx, mvy, mspd, dx, dy, cnt

    cx_o, cy_o, mvx_o, mvy_o, spd_o, dx_o, dy_o, cnt_o = _team_stats(vp_own)
    cx_x, cy_x, mvx_x, mvy_x, spd_x, dx_x, dy_x, cnt_x = _team_stats(vp_opp)

    ax = cx_x - cx_o
    ay = cy_x - cy_o
    norm = np.maximum(np.hypot(ax, ay), 2.0)
    ux = (ax / norm).astype(np.float32)
    uy = (ay / norm).astype(np.float32)
    # Orthogonal across-axis magnitude doesn't depend on handedness sign
    qx = -uy
    qy = ux

    # Average signed player directional agreement along centroid axis
    def _agree(v_mask, ux_dir, uy_dir):
        v_len = np.maximum(spd_p, 0.5)
        ok = v_mask & (spd_p > 0.5)
        cos_t = np.where(ok, (vx * ux_dir[:, None] + vy * uy_dir[:, None]) / v_len, 0.0)
        denom = np.maximum(ok.sum(axis=1).astype(np.float32), 1.0)
        return (cos_t.sum(axis=1) / denom).astype(np.float32)

    agree_o = _agree(vp_own, ux, uy)
    agree_x = _agree(vp_opp, -ux, -uy)

    # Spread along opposing axis
    spr_o = np.sqrt(np.maximum((np.where(vp_own, (dx_o * ux[:, None] + dy_o * uy[:, None]) ** 2, 0.0)).sum(axis=1) / cnt_o, 0.0)).astype(np.float32)
    spr_x = np.sqrt(np.maximum((np.where(vp_opp, (dx_x * ux[:, None] + dy_x * uy[:, None]) ** 2, 0.0)).sum(axis=1) / cnt_x, 0.0)).astype(np.float32)

    # Block-respecting consecutive W-frame centroid displacement velocity & speed/spread change
    feat_cols = []
    for W in (10, 30):
        vcx_o = np.zeros(n, dtype=np.float32)
        vcy_o = np.zeros(n, dtype=np.float32)
        vcx_x = np.zeros(n, dtype=np.float32)
        vcy_x = np.zeros(n, dtype=np.float32)
        dsc_o = np.zeros(n, dtype=np.float32)
        dsc_x = np.zeros(n, dtype=np.float32)
        dspr_o = np.zeros(n, dtype=np.float32)
        dspr_x = np.zeros(n, dtype=np.float32)
        for b0, b1 in m.blocks:
            L = b1 - b0
            if L <= 1:
                continue
            step = min(W, L - 1)
            if L > step:
                dt = float(step) / FPS
                vcx_o[b0 + step : b1] = (cx_o[b0 + step : b1] - cx_o[b0 : b1 - step]) / dt
                vcy_o[b0 + step : b1] = (cy_o[b0 + step : b1] - cy_o[b0 : b1 - step]) / dt
                vcx_x[b0 + step : b1] = (cx_x[b0 + step : b1] - cx_x[b0 : b1 - step]) / dt
                vcy_x[b0 + step : b1] = (cy_x[b0 + step : b1] - cy_x[b0 : b1 - step]) / dt
                dsc_o[b0 + step : b1] = spd_o[b0 + step : b1] - spd_o[b0 : b1 - step]
                dsc_x[b0 + step : b1] = spd_x[b0 + step : b1] - spd_x[b0 : b1 - step]
                dspr_o[b0 + step : b1] = spr_o[b0 + step : b1] - spr_o[b0 : b1 - step]
                dspr_x[b0 + step : b1] = spr_x[b0 + step : b1] - spr_x[b0 : b1 - step]

        va_o = vcx_o * ux + vcy_o * uy
        va_x = vcx_x * (-ux) + vcy_x * (-uy)
        vac_o = np.abs(vcx_o * qx + vcy_o * qy)
        vac_x = np.abs(vcx_x * qx + vcy_x * qy)
        # Blend instantaneous player mean velocity along axis for W=10 responsiveness
        if W == 10:
            vmean_o = mvx_o * ux + mvy_o * uy
            vmean_x = mvx_x * (-ux) + mvy_x * (-uy)
            feat_cols.extend([
                0.5 * va_o + 0.5 * vmean_o,
                0.5 * va_x + 0.5 * vmean_x,
                (0.5 * va_o + 0.5 * vmean_o) - (0.5 * va_x + 0.5 * vmean_x),
                agree_o,
                agree_x,
                agree_o - agree_x,
                dsc_o - dsc_x,
            ])
        else:
            feat_cols.extend([
                va_o,
                va_x,
                va_o - va_x,
                vac_o - vac_x,
                agree_o - agree_x,
                dsc_o - dsc_x,
                dspr_o,
                dspr_o - dspr_x,
            ])

    base15 = np.nan_to_num(np.column_stack(feat_cols), nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)
    r10 = _roll_mean_blocks(base15, m.blocks, 10)
    r30 = _roll_mean_blocks(base15, m.blocks, 30)
    return np.hstack([base15, r10, r30]).astype(np.float32)


def compute_deblurred_ball_features(
    m: Match,
    rb: ReconstructedBall,
    px: np.ndarray,
    py: np.ndarray,
    vp_own: np.ndarray,
    vp_opp: np.ndarray,
) -> np.ndarray:
    """Laplacian de-blurred ball kinematics + micro-touch kinks (10 cols)."""
    n = m.n
    bxy = rb.ball_hat.astype(np.float32, copy=False)
    bconf = rb.ball_conf.astype(np.float32, copy=False)
    sm18 = _roll_mean_blocks(bxy, m.blocks, 9)
    bsharp = bxy - 0.35 * (sm18 - bxy)
    bx_s, by_s = bsharp[:, 0], bsharp[:, 1]

    d2_own = np.where(vp_own, (px - bx_s[:, None]) ** 2 + (py - by_s[:, None]) ** 2, 1e9)
    d2_opp = np.where(vp_opp, (px - bx_s[:, None]) ** 2 + (py - by_s[:, None]) ** 2, 1e9)
    d1_own_sh = np.sqrt(np.min(d2_own, axis=1)).astype(np.float32) if d2_own.shape[1] else np.full(n, 40.0, dtype=np.float32)
    d1_opp_sh = np.sqrt(np.min(d2_opp, axis=1)).astype(np.float32) if d2_opp.shape[1] else np.full(n, 40.0, dtype=np.float32)
    d1_sh_diff = (d1_opp_sh - d1_own_sh) * bconf

    def _kink(step: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        for b0, b1 in m.blocks:
            if b1 - b0 <= 2 * step:
                continue
            dx1 = bx_s[b0 + step : b1 - step] - bx_s[b0 : b1 - 2 * step]
            dy1 = by_s[b0 + step : b1 - step] - by_s[b0 : b1 - 2 * step]
            dx2 = bx_s[b0 + 2 * step : b1] - bx_s[b0 + step : b1 - step]
            dy2 = by_s[b0 + 2 * step : b1] - by_s[b0 + step : b1 - step]
            n1 = np.maximum(np.hypot(dx1, dy1), 1e-3)
            n2 = np.maximum(np.hypot(dx2, dy2), 1e-3)
            cos_a = np.clip((dx1 * dx2 + dy1 * dy2) / (n1 * n2), -1.0, 1.0)
            spd_min = np.minimum(n1, n2) * (FPS / float(step))
            out[b0 + step : b1 - step] = 0.5 * (1.0 - cos_a) * np.minimum(spd_min, 15.0)
        return out * bconf

    k10 = _kink(10)
    k20 = _kink(20)
    base5 = np.nan_to_num(
        np.column_stack([d1_own_sh, d1_opp_sh, d1_sh_diff, k10, k20]),
        nan=0.0,
        posinf=40.0,
        neginf=-40.0,
    ).astype(np.float32)
    r10 = _roll_mean_blocks(base5, m.blocks, 10)
    return np.hstack([base5, r10]).astype(np.float32)


def ball_free_team_cols(n_cols: int) -> Optional[np.ndarray]:
    """Return strictly ball-free equivariant column indices for gap specialist model."""
    if n_cols >= 201:
        return np.arange(156, 201, dtype=np.int64)
    return None
