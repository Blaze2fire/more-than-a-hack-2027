"""Algorithmic ball trajectory cleaning, Hermite interpolation, and constellation imputation.

Purely physics- and player-geometry-driven reconstruction (no competition GT):
1. Remove pitch-out-of-bounds and high-velocity teleport glitches (>38 m/s).
2. Interpolate short/medium dropouts with velocity-damped cubic Hermite splines.
3. Learn a team-symmetric player-constellation residual regressor around the local
   player-density center to localize the ball during longer burst gaps (>1-30s).
4. Smoothly blend clean observations, Hermite kinematics, and constellation estimates
   and return (ball_hat, vis_raw, vis_clean, ball_conf, gap_len, ball_vel, ball_acc).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from solution.common import FPS, Match
from solution.model import _smooth_nan

MAX_BALL_JUMP_SPD = 38.0
MAX_HERMITE_VEL = 26.0
CHUNK_FRAMES = 4096


def compute_player_vel(pos: np.ndarray, blocks: List[Tuple[int, int]], max_spd: float = 12.0) -> np.ndarray:
    """Smoothed (n, S, 2) m/s velocity vectors across contiguous blocks."""
    n, s, _ = pos.shape
    vel = np.full((n, s, 2), np.nan, dtype=np.float32)
    for b0, b1 in blocks:
        if b1 - b0 < 3:
            continue
        p = pos[b0:b1].astype(np.float64, copy=False)
        valid = np.isfinite(p[:, :, 0]) & np.isfinite(p[:, :, 1])
        sm = p.copy()
        mid_ok = valid[1:-1] & valid[:-2] & valid[2:]
        sm[1:-1] = np.where(mid_ok[:, :, None], 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:], p[1:-1])

        vx = np.full((b1 - b0, s), np.nan, dtype=np.float64)
        vy = np.full((b1 - b0, s), np.nan, dtype=np.float64)
        c2 = valid[:-4] & valid[4:]
        vx[2:-2] = np.where(c2, (sm[4:, :, 0] - sm[:-4, :, 0]) * (FPS / 4.0), vx[2:-2])
        vy[2:-2] = np.where(c2, (sm[4:, :, 1] - sm[:-4, :, 1]) * (FPS / 4.0), vy[2:-2])

        c1 = np.isnan(vx[1:-1]) & valid[:-2] & valid[2:]
        vx[1:-1] = np.where(c1, (sm[2:, :, 0] - sm[:-2, :, 0]) * (FPS / 2.0), vx[1:-1])
        vy[1:-1] = np.where(c1, (sm[2:, :, 1] - sm[:-2, :, 1]) * (FPS / 2.0), vy[1:-1])

        spd = np.hypot(vx, vy)
        too_fast = spd > max_spd * 1.4
        vx[too_fast] = np.nan
        vy[too_fast] = np.nan
        scale = np.where(spd > max_spd, max_spd / np.maximum(spd, 1e-6), 1.0)
        vel[b0:b1, :, 0] = (vx * scale).astype(np.float32)
        vel[b0:b1, :, 1] = (vy * scale).astype(np.float32)
    return vel


def clean_raw_ball(
    ball: np.ndarray,
    blocks: List[Tuple[int, int]],
    pitch: Tuple[float, float],
    max_jump_spd: float = MAX_BALL_JUMP_SPD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject out-of-bounds glitches and short teleport spikes (>38 m/s)."""
    n = len(ball)
    vis_raw = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    clean = ball.copy().astype(np.float32)
    pitch_l, pitch_w = pitch
    oob = vis_raw & ((clean[:, 0] < -8.0) | (clean[:, 0] > pitch_l + 8.0) | (clean[:, 1] < -8.0) | (clean[:, 1] > pitch_w + 8.0))
    clean[oob] = np.nan

    for b0, b1 in blocks:
        sub = clean[b0:b1]
        v_idx = np.where(np.isfinite(sub[:, 0]))[0]
        if len(v_idx) < 3:
            continue
        # Pass: drop isolated 1-3 frame glitches whose jump velocity exceeds max_jump_spd
        # while the surrounding anchor points connect physically cleanly.
        keep = np.ones(len(v_idx), dtype=bool)
        # First mark pairs jumping faster than physical ball speed across short intervals.
        for k in range(1, len(v_idx) - 1):
            if not keep[k]:
                continue
            prev_k = k - 1
            while prev_k > 0 and not keep[prev_k]:
                prev_k -= 1
            t0, t1 = v_idx[prev_k], v_idx[k]
            dt01 = max(1, t1 - t0) / FPS
            d01 = float(np.hypot(sub[t1, 0] - sub[t0, 0], sub[t1, 1] - sub[t0, 1]))
            if d01 / dt01 > max_jump_spd:
                # Look ahead up to 4 observations to find where trajectory returns near t0
                found_rejoin = False
                for m_step in range(1, min(5, len(v_idx) - k)):
                    t2 = v_idx[k + m_step]
                    dt02 = max(1, t2 - t0) / FPS
                    d02 = float(np.hypot(sub[t2, 0] - sub[t0, 0], sub[t2, 1] - sub[t0, 1]))
                    if d02 / dt02 <= max_jump_spd:
                        keep[k : k + m_step] = False
                        found_rejoin = True
                        break
                if not found_rejoin and k == 1:
                    # Handle leading glitch at first frame
                    dt12 = max(1, v_idx[2] - v_idx[1]) / FPS
                    d12 = float(np.hypot(sub[v_idx[2], 0] - sub[v_idx[1], 0], sub[v_idx[2], 1] - sub[v_idx[1], 1]))
                    if d12 / dt12 <= max_jump_spd:
                        keep[0] = False
        dropped = v_idx[~keep]
        if len(dropped):
            sub[dropped] = np.nan

    vis_clean = np.isfinite(clean[:, 0]) & np.isfinite(clean[:, 1])
    return clean, vis_raw, vis_clean


def _estimate_ball_velocities(clean_ball: np.ndarray, b0: int, b1: int) -> np.ndarray:
    """Instantaneous 2D ball velocity over clean consecutive observations inside block."""
    sub = clean_ball[b0:b1]
    nb = b1 - b0
    v = np.zeros((nb, 2), dtype=np.float32)
    ok = np.isfinite(sub[:, 0])
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return v
    # Finite diff between adjacent valid observations (only if <= 5 frames apart)
    dt = np.diff(idx)
    good = dt <= 5
    dx = np.diff(sub[idx, 0]) / (dt / FPS)
    dy = np.diff(sub[idx, 1]) / (dt / FPS)
    dx = np.where(good, np.clip(dx, -MAX_HERMITE_VEL, MAX_HERMITE_VEL), 0.0)
    dy = np.where(good, np.clip(dy, -MAX_HERMITE_VEL, MAX_HERMITE_VEL), 0.0)
    # Assign weighted average at idx nodes and forward-fill short spans
    vx_sparse = np.full(nb, np.nan, dtype=np.float32)
    vy_sparse = np.full(nb, np.nan, dtype=np.float32)
    vx_sparse[idx[:-1]] = dx
    vy_sparse[idx[:-1]] = dy
    vx_sparse[idx[-1]] = dx[-1]
    vy_sparse[idx[-1]] = dy[-1]
    v[:, 0] = np.nan_to_num(_smooth_nan(vx_sparse, 2), nan=0.0)
    v[:, 1] = np.nan_to_num(_smooth_nan(vy_sparse, 2), nan=0.0)
    return np.clip(v, -MAX_HERMITE_VEL, MAX_HERMITE_VEL)


def extract_constellation_features(m: Match, vel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute team-symmetric player-constellation anchor coordinates and feature bank.

    Returns:
      anchor_xy: (n, 2) metres (local player-density peak, smooth fallback to median)
      feats:     (n, 29) float32 team-symmetric features around anchor_xy & pitch
    """
    n, s, _ = m.pos.shape
    pitch_l, pitch_w = m.pitch
    if vel is None:
        vel = compute_player_vel(m.pos, m.blocks)

    px = m.pos[:, :, 0].astype(np.float32, copy=False)
    py = m.pos[:, :, 1].astype(np.float32, copy=False)
    vx = np.nan_to_num(vel[:, :, 0], nan=0.0).astype(np.float32, copy=False)
    vy = np.nan_to_num(vel[:, :, 1], nan=0.0).astype(np.float32, copy=False)
    spd = np.nan_to_num(m.spd, nan=0.0).astype(np.float32, copy=False)
    valid = np.isfinite(px) & np.isfinite(py)
    t_arr = m.team.astype(np.int8)
    same_team = (t_arr[:, None] == t_arr[None, :])  # (S, S)

    # Pre-allocate anchor components
    med_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    med_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    dens_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    dens_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    dens4_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    dens4_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    duel_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    duel_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    duel3_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    duel3_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    oppw_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    oppw_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    spdw_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    spdw_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)
    conv_x = np.full(n, 0.5 * pitch_l, dtype=np.float32)
    conv_y = np.full(n, 0.5 * pitch_w, dtype=np.float32)

    std_x = np.full(n, 15.0, dtype=np.float32)
    std_y = np.full(n, 12.0, dtype=np.float32)
    min_duel_d = np.full(n, 10.0, dtype=np.float32)
    mean_spd = np.zeros(n, dtype=np.float32)
    max_spd = np.zeros(n, dtype=np.float32)
    team_sep_x = np.zeros(n, dtype=np.float32)
    team_sep_y = np.zeros(n, dtype=np.float32)
    team_comp_min = np.full(n, 12.0, dtype=np.float32)
    team_comp_max = np.full(n, 18.0, dtype=np.float32)

    th_mask = (t_arr == 0)[None, :]
    ta_mask = (t_arr == 1)[None, :]

    for start in range(0, n, CHUNK_FRAMES):
        end = min(n, start + CHUNK_FRAMES)
        m_len = end - start
        v_c = valid[start:end]  # (B, S)
        has_any = v_c.any(axis=1)
        px_c = np.where(v_c, px[start:end], np.nan)
        py_c = np.where(v_c, py[start:end], np.nan)
        vx_c = np.where(v_c, vx[start:end], 0.0)
        vy_c = np.where(v_c, vy[start:end], 0.0)
        sp_c = np.where(v_c, spd[start:end], 0.0)

        # All-player median & dispersion (safe on all-NaN frames)
        if has_any.all():
            mx = np.nanmedian(px_c, axis=1)
            my = np.nanmedian(py_c, axis=1)
            sx = np.nanstd(px_c, axis=1)
            sy = np.nanstd(py_c, axis=1)
        else:
            mx = np.full(m_len, 0.5 * pitch_l, dtype=np.float32)
            my = np.full(m_len, 0.5 * pitch_w, dtype=np.float32)
            sx = np.full(m_len, 15.0, dtype=np.float32)
            sy = np.full(m_len, 12.0, dtype=np.float32)
            if has_any.any():
                mx[has_any] = np.nanmedian(px_c[has_any], axis=1)
                my[has_any] = np.nanmedian(py_c[has_any], axis=1)
                sx[has_any] = np.nanstd(px_c[has_any], axis=1)
                sy[has_any] = np.nanstd(py_c[has_any], axis=1)
        med_x[start:end] = mx
        med_y[start:end] = my
        std_x[start:end] = np.nan_to_num(sx, nan=15.0)
        std_y[start:end] = np.nan_to_num(sy, nan=12.0)

        # Team-symmetric centroids and compactnesses (always |H - A| / min / max)
        vh = v_c & th_mask
        va = v_c & ta_mask
        cnt_h = np.maximum(vh.sum(axis=1), 1)
        cnt_a = np.maximum(va.sum(axis=1), 1)
        hx = np.where(vh, np.nan_to_num(px_c), 0.0).sum(axis=1) / cnt_h
        hy = np.where(vh, np.nan_to_num(py_c), 0.0).sum(axis=1) / cnt_h
        ax = np.where(va, np.nan_to_num(px_c), 0.0).sum(axis=1) / cnt_a
        ay = np.where(va, np.nan_to_num(py_c), 0.0).sum(axis=1) / cnt_a
        team_sep_x[start:end] = np.abs(hx - ax)
        team_sep_y[start:end] = np.abs(hy - ay)

        ch_std = np.sqrt(
            (np.where(vh, (np.nan_to_num(px_c) - hx[:, None]) ** 2 + (np.nan_to_num(py_c) - hy[:, None]) ** 2, 0.0).sum(axis=1))
            / cnt_h
        )
        ca_std = np.sqrt(
            (np.where(va, (np.nan_to_num(px_c) - ax[:, None]) ** 2 + (np.nan_to_num(py_c) - ay[:, None]) ** 2, 0.0).sum(axis=1))
            / cnt_a
        )
        team_comp_min[start:end] = np.minimum(ch_std, ca_std)
        team_comp_max[start:end] = np.maximum(ch_std, ca_std)

        cnt_all = np.maximum(v_c.sum(axis=1), 1)
        mean_spd[start:end] = sp_c.sum(axis=1) / cnt_all
        max_spd[start:end] = sp_c.max(axis=1)

        # Exclude deep goalkeepers when computing velocity & speed weights (within 5m of endlines)
        outfield = v_c & (np.nan_to_num(px_c, nan=-999.0) > 6.0) & (np.nan_to_num(px_c, nan=-999.0) < pitch_l - 6.0)
        w_spd = np.where(outfield, sp_c ** 2 + 0.05, np.where(v_c, 0.01, 0.0))
        w_spd_sum = np.maximum(w_spd.sum(axis=1), 1e-6)
        px0 = np.nan_to_num(px_c, nan=0.0)
        py0 = np.nan_to_num(py_c, nan=0.0)
        spdw_x[start:end] = (w_spd * px0).sum(axis=1) / w_spd_sum
        spdw_y[start:end] = (w_spd * py0).sum(axis=1) / w_spd_sum

        # Forward velocity projection focus (p + 0.6*v weighted by speed)
        w_conv = np.where(v_c, np.maximum(sp_c - 0.5, 0.05) ** 1.5, 0.0)
        w_conv_sum = np.maximum(w_conv.sum(axis=1), 1e-6)
        conv_x[start:end] = (w_conv * (px0 + 0.6 * vx_c)).sum(axis=1) / w_conv_sum
        conv_y[start:end] = (w_conv * (py0 + 0.6 * vy_c)).sum(axis=1) / w_conv_sum

        # Pairwise distances (B, S, S)
        both_v = v_c[:, :, None] & v_c[:, None, :]
        dx = px0[:, :, None] - px0[:, None, :]
        dy = py0[:, :, None] - py0[:, None, :]
        d2 = np.where(both_v, dx * dx + dy * dy, np.float32(1e6))

        # Local player density peaks (sigma=8m and sigma=4m, excluding self)
        eye = np.eye(s, dtype=bool)[None, :, :]
        d2_noself = np.where(eye, np.float32(1e6), d2)
        k8 = np.exp(-d2_noself / np.float32(2.0 * 8.0 * 8.0)).sum(axis=2)
        w8 = np.where(v_c, k8 ** 2 + np.float32(1e-4), np.float32(0.0))
        w8s = np.maximum(w8.sum(axis=1), 1e-6)
        dens_x[start:end] = (w8 * px0).sum(axis=1) / w8s
        dens_y[start:end] = (w8 * py0).sum(axis=1) / w8s

        k4 = np.exp(-d2_noself / np.float32(2.0 * 4.5 * 4.5)).sum(axis=2)
        w4 = np.where(v_c, k4 ** 2 + np.float32(1e-4), np.float32(0.0))
        w4s = np.maximum(w4.sum(axis=1), 1e-6)
        dens4_x[start:end] = (w4 * px0).sum(axis=1) / w4s
        dens4_y[start:end] = (w4 * py0).sum(axis=1) / w4s

        # Nearest opponent distance per player & closest inter-team duel pairs
        opp_mask = both_v & (~same_team[None, :, :])
        d2_opp = np.where(opp_mask, d2, np.float32(1e6))
        min_opp_d2 = d2_opp.min(axis=2)  # (B, S)
        w_opp = np.where(v_c, 1.0 / (min_opp_d2 + np.float32(6.0)) ** 1.5, np.float32(0.0))
        w_opp_s = np.maximum(w_opp.sum(axis=1), 1e-6)
        oppw_x[start:end] = (w_opp * px0).sum(axis=1) / w_opp_s
        oppw_y[start:end] = (w_opp * py0).sum(axis=1) / w_opp_s

        # Closest single opponent pair midpoint and top-3 tightest opposing pairs
        d2_opp_flat = d2_opp.reshape(m_len, -1)
        idx_min = np.argmin(d2_opp_flat, axis=1)
        i_min = idx_min // s
        j_min = idx_min % s
        b_arange = np.arange(m_len)
        duel_x[start:end] = 0.5 * (px0[b_arange, i_min] + px0[b_arange, j_min])
        duel_y[start:end] = 0.5 * (py0[b_arange, i_min] + py0[b_arange, j_min])
        min_duel_d[start:end] = np.sqrt(np.clip(d2_opp_flat[b_arange, idx_min], 0.0, 2500.0))

        # Top-3 tightest opposing pairs weighted average (via top-6 asymmetric entries)
        k_top = min(6, s * s)
        part_idx = np.argpartition(d2_opp_flat, k_top - 1, axis=1)[:, :k_top]
        top_d2 = d2_opp_flat[b_arange[:, None], part_idx]
        w_pair = 1.0 / (top_d2 + np.float32(4.0))
        wp_sum = np.maximum(w_pair.sum(axis=1), 1e-6)
        pi = part_idx // s
        pj = part_idx % s
        mid_x = 0.5 * (px0[b_arange[:, None], pi] + px0[b_arange[:, None], pj])
        mid_y = 0.5 * (py0[b_arange[:, None], pi] + py0[b_arange[:, None], pj])
        duel3_x[start:end] = (w_pair * mid_x).sum(axis=1) / wp_sum
        duel3_y[start:end] = (w_pair * mid_y).sum(axis=1) / wp_sum

    # Smooth anchors slightly over time to suppress single-frame swapping jitter
    anchor_x = dens_x.copy()
    anchor_y = dens_y.copy()
    for b0, b1 in m.blocks:
        anchor_x[b0:b1] = np.nan_to_num(_smooth_nan(anchor_x[b0:b1], 3), nan=0.5 * pitch_l)
        anchor_y[b0:b1] = np.nan_to_num(_smooth_nan(anchor_y[b0:b1], 3), nan=0.5 * pitch_w)

    # Multi-scale temporal smoothed deltas relative to anchor (all normalized by pitch/scale)
    inv_l = 1.0 / pitch_l
    inv_w = 1.0 / pitch_w
    feats_list = [
        (anchor_x - 0.5 * pitch_l) * inv_l,
        (anchor_y - 0.5 * pitch_w) * inv_w,
        (med_x - anchor_x) * inv_l,
        (med_y - anchor_y) * inv_w,
        (dens4_x - anchor_x) * inv_l,
        (dens4_y - anchor_y) * inv_w,
        (duel_x - anchor_x) * inv_l,
        (duel_y - anchor_y) * inv_w,
        (duel3_x - anchor_x) * inv_l,
        (duel3_y - anchor_y) * inv_w,
        (oppw_x - anchor_x) * inv_l,
        (oppw_y - anchor_y) * inv_w,
        (spdw_x - anchor_x) * inv_l,
        (spdw_y - anchor_y) * inv_w,
        (conv_x - anchor_x) * inv_l,
        (conv_y - anchor_y) * inv_w,
        std_x * inv_l,
        std_y * inv_w,
        min_duel_d,
        mean_spd,
        max_spd,
        team_sep_x * inv_l,
        team_sep_y * inv_w,
        team_comp_min * inv_l,
        team_comp_max * inv_l,
    ]
    # Add short/medium temporal velocities of anchor & oppw within blocks
    ax_s12 = anchor_x.copy()
    ay_s12 = anchor_y.copy()
    ox_s25 = oppw_x.copy()
    oy_s25 = oppw_y.copy()
    for b0, b1 in m.blocks:
        ax_s12[b0:b1] = _smooth_nan(anchor_x[b0:b1], 12)
        ay_s12[b0:b1] = _smooth_nan(anchor_y[b0:b1], 12)
        ox_s25[b0:b1] = _smooth_nan(opp_xy := oppw_x[b0:b1], 25)
        oy_s25[b0:b1] = _smooth_nan(oppw_y[b0:b1], 25)
    feats_list.extend([
        (ax_s12 - anchor_x) * inv_l,
        (ay_s12 - anchor_y) * inv_w,
        (ox_s25 - anchor_x) * inv_l,
        (oy_s25 - anchor_y) * inv_w,
    ])
    X_feat = np.column_stack(feats_list).astype(np.float32)
    anchor_xy = np.column_stack([anchor_x, anchor_y]).astype(np.float32)
    return anchor_xy, X_feat


@dataclass
class BallConstellationModel:
    """Pair of HGB regressors predicting (ball - anchor) normalised to pitch size."""
    reg_x: HistGradientBoostingRegressor
    reg_y: HistGradientBoostingRegressor

    def predict(self, m: Match, vel: Optional[np.ndarray] = None) -> np.ndarray:
        anchor_xy, X_feat = extract_constellation_features(m, vel=vel)
        pitch_l, pitch_w = m.pitch
        dx_norm = self.reg_x.predict(X_feat).astype(np.float32)
        dy_norm = self.reg_y.predict(X_feat).astype(np.float32)
        px = np.clip(anchor_xy[:, 0] + dx_norm * pitch_l, -2.0, pitch_l + 2.0)
        py = np.clip(anchor_xy[:, 1] + dy_norm * pitch_w, -2.0, pitch_w + 2.0)
        return np.column_stack([px, py]).astype(np.float32)


def fit_constellation_imputer(
    matches: List[Match],
    stride: int = 5,
    seed: int = 0,
) -> BallConstellationModel:
    """Fit player-constellation ball imputer on external matches with visible ball."""
    Xs, yxs, yys = [], [], []
    for m in matches:
        clean_b, _, vis = clean_raw_ball(m.ball, m.blocks, m.pitch)
        anchor_xy, X_feat = extract_constellation_features(m)
        pitch_l, pitch_w = m.pitch
        idx_mask = np.zeros(m.n, dtype=bool)
        for b0, b1 in m.blocks:
            idx_mask[b0:b1:stride] = True
        in_pitch = (
            (clean_b[:, 0] >= -1.0) & (clean_b[:, 0] <= pitch_l + 1.0)
            & (clean_b[:, 1] >= -1.0) & (clean_b[:, 1] <= pitch_w + 1.0)
        )
        live_ok = (m.gt == 0) | (m.gt == 1) if m.gt is not None else np.ones(m.n, dtype=bool)
        idx_mask &= vis & in_pitch & live_ok
        if not idx_mask.any():
            continue
        dx = np.clip((clean_b[idx_mask, 0] - anchor_xy[idx_mask, 0]) / pitch_l, -0.45, 0.45)
        dy = np.clip((clean_b[idx_mask, 1] - anchor_xy[idx_mask, 1]) / pitch_w, -0.45, 0.45)
        Xs.append(X_feat[idx_mask])
        yxs.append(dx.astype(np.float32))
        yys.append(dy.astype(np.float32))

    X_all = np.vstack(Xs)
    yx_all = np.concatenate(yxs)
    yy_all = np.concatenate(yys)

    kw = dict(
        max_iter=160,
        max_depth=6,
        learning_rate=0.06,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    reg_x = HistGradientBoostingRegressor(**kw).fit(X_all, yx_all)
    reg_y = HistGradientBoostingRegressor(**kw).fit(X_all, yy_all)
    return BallConstellationModel(reg_x=reg_x, reg_y=reg_y)


@dataclass
class ReconstructedBall:
    ball_hat: np.ndarray        # (n, 2) continuous reconstructed ball metres
    vis_raw: np.ndarray         # (n,) bool raw observation present
    vis_clean: np.ndarray       # (n,) bool clean observation after glitch removal
    ball_conf: np.ndarray       # (n,) float32 in [0, 1] localization reliability
    gap_len: np.ndarray         # (n,) float32 frames in current missing burst (0 if vis_clean)
    dist_to_vis: np.ndarray     # (n,) float32 frames to nearest clean visible observation
    ball_vel: np.ndarray        # (n, 2) float32 reconstructed velocity m/s
    ball_spd: np.ndarray        # (n,) float32 reconstructed speed m/s
    ball_acc: np.ndarray        # (n,) float32 magnitude of velocity change m/s^2
    const_xy: np.ndarray        # (n, 2) float32 constellation ball estimate


def reconstruct_ball(
    m: Match,
    imputer: Optional[BallConstellationModel] = None,
    vel: Optional[np.ndarray] = None,
) -> ReconstructedBall:
    """Clean, Hermite-interpolate, and constellation-impute the ball trajectory."""
    n = m.n
    pitch_l, pitch_w = m.pitch
    clean_b, vis_raw, vis_clean = clean_raw_ball(m.ball, m.blocks, m.pitch)
    if imputer is not None:
        const_xy = imputer.predict(m, vel=vel)
    else:
        const_xy, _ = extract_constellation_features(m, vel=vel)

    ball_hat = const_xy.copy()
    ball_conf = np.full(n, 0.20, dtype=np.float32)
    gap_len = np.zeros(n, dtype=np.float32)
    dist_to_vis = np.full(n, 500.0, dtype=np.float32)

    for b0, b1 in m.blocks:
        sub_c = clean_b[b0:b1]
        sub_vis = vis_clean[b0:b1]
        sub_const = const_xy[b0:b1]
        nb = b1 - b0
        out = sub_const.copy()
        conf = np.full(nb, 0.22, dtype=np.float32)
        dvis = np.full(nb, 500.0, dtype=np.float32)
        glen = np.zeros(nb, dtype=np.float32)

        vis_idx = np.where(sub_vis)[0]
        if len(vis_idx) == 0:
            # A fully hidden block is one long gap, not a sequence of zero gaps.
            # The optional 2-12s ball-free router relies on this distinction.
            glen[:] = nb
        if len(vis_idx) > 0:
            v_est = _estimate_ball_velocities(clean_b, b0, b1)
            out[vis_idx] = sub_c[vis_idx]
            conf[vis_idx] = 1.0
            dvis[vis_idx] = 0.0

            # Leading gap [0, vis_idx[0])
            s0 = int(vis_idx[0])
            if s0 > 0:
                glen[0:s0] = s0
                d_edge = np.arange(s0, 0, -1, dtype=np.float32)
                dvis[0:s0] = d_edge
                w = np.exp(-d_edge / 10.0)[:, None]
                out[0:s0] = w * sub_c[s0] + (1.0 - w) * sub_const[0:s0]
                conf[0:s0] = np.maximum(0.22, 0.85 * w[:, 0])

            # Trailing gap [vis_idx[-1] + 1, nb)
            e_last = int(vis_idx[-1]) + 1
            if e_last < nb:
                L = nb - e_last
                glen[e_last:nb] = L
                d_edge = np.arange(1, L + 1, dtype=np.float32)
                dvis[e_last:nb] = d_edge
                w = np.exp(-d_edge / 10.0)[:, None]
                out[e_last:nb] = w * sub_c[e_last - 1] + (1.0 - w) * sub_const[e_last:nb]
                conf[e_last:nb] = np.maximum(0.22, 0.85 * w[:, 0])

            # Internal gaps between consecutive clean visible points
            gap_starts = np.where(np.diff(vis_idx) > 1)[0]
            for gi in gap_starts:
                a_i = int(vis_idx[gi])
                b_i = int(vis_idx[gi + 1])
                s = a_i + 1
                L = b_i - s
                glen[s:b_i] = L
                t_steps = np.arange(1, L + 1, dtype=np.float32)
                u = t_steps / float(L + 1)
                d_nearest = np.minimum(t_steps, float(L + 1) - t_steps)
                dvis[s:b_i] = d_nearest

                p0 = sub_c[a_i]
                p1 = sub_c[b_i]
                dt_s = float(b_i - a_i) / FPS
                chord_spd = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / dt_s

                # Velocity-damped cubic Hermite spline (exact on <=2s gaps; damped chord->constellation on longer bursts)
                damp = float(np.exp(-L / 35.0)) * (1.0 if chord_spd < 24.0 else 0.25)
                m0 = v_est[a_i] * dt_s * damp
                m1 = v_est[b_i] * dt_s * damp
                h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
                h10 = u**3 - 2.0 * u**2 + u
                h01 = -2.0 * u**3 + 3.0 * u**2
                h11 = u**3 - u**2
                p_herm = (
                    h00[:, None] * p0
                    + h10[:, None] * m0
                    + h01[:, None] * p1
                    + h11[:, None] * m1
                )
                p_herm[:, 0] = np.clip(p_herm[:, 0], -3.0, pitch_l + 3.0)
                p_herm[:, 1] = np.clip(p_herm[:, 1], -3.0, pitch_w + 3.0)

                if L <= 50:
                    w_kin = np.ones(L, dtype=np.float32)
                    out[s:b_i] = p_herm
                else:
                    # Smoothly hand off from Hermite kinematics (first 0.8s inside gap) to constellation + weak chord prior
                    w_kin = np.exp(-((np.maximum(0.0, d_nearest - 15.0) / 30.0) ** 1.4))
                    w_end = np.exp(-d_nearest / 110.0)[:, None]
                    p_far = 0.35 * w_end * ((1.0 - u[:, None]) * p0 + u[:, None] * p1) + (1.0 - 0.35 * w_end) * sub_const[s:b_i]
                    out[s:b_i] = w_kin[:, None] * p_herm + (1.0 - w_kin[:, None]) * p_far
                conf[s:b_i] = np.clip(0.22 + 0.72 * np.exp(-d_nearest / 25.0), 0.22, 0.96)

        ball_hat[b0:b1] = out
        ball_conf[b0:b1] = conf
        gap_len[b0:b1] = glen
        dist_to_vis[b0:b1] = dvis

    # Reconstructed velocity and acceleration across contiguous blocks
    bvel = np.zeros((n, 2), dtype=np.float32)
    bspd = np.zeros(n, dtype=np.float32)
    bacc = np.zeros(n, dtype=np.float32)
    for b0, b1 in m.blocks:
        if b1 - b0 < 5:
            continue
        p = ball_hat[b0:b1]
        v = np.zeros((b1 - b0, 2), dtype=np.float32)
        v[2:-2] = (p[4:] - p[:-4]) * (FPS / 4.0)
        v[0:2] = v[2:3]
        v[-2:] = v[-3:-2]
        sp = np.hypot(v[:, 0], v[:, 1])
        # Acceleration vector norm over +/-2 frames (detects kicks, deflections, tackles!)
        dv = np.zeros((b1 - b0, 2), dtype=np.float32)
        dv[2:-2] = (v[4:] - v[:-4]) * (FPS / 4.0)
        acc = np.hypot(dv[:, 0], dv[:, 1])
        bvel[b0:b1] = v
        bspd[b0:b1] = np.clip(sp, 0.0, 40.0)
        bacc[b0:b1] = np.clip(acc, 0.0, 120.0)

    return ReconstructedBall(
        ball_hat=ball_hat,
        vis_raw=vis_raw,
        vis_clean=vis_clean,
        ball_conf=ball_conf,
        gap_len=gap_len,
        dist_to_vis=dist_to_vis,
        ball_vel=bvel,
        ball_spd=bspd,
        ball_acc=bacc,
        const_xy=const_xy,
    )
