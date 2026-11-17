"""Equivariant PlayerSet-TCN + BiGRU sequence model for live team ownership & changepoints.

Design guarantees:
1. Strict zero dead-ball leakage (dead_leak == 0.0): this module predicts ONLY
   conditional live team ownership P(own | live) and live H<->A changepoint P(switch | live).
   It never modifies P(dead).
2. Bit-for-bit H<->A swap equivariance (swap_diff == 0.0): evaluated symmetrically on both
   (own, opp, Xtf) and (opp, own, Xts) orientations, combined via normalized difference
   d_ha = (p_f - p_s) / max(p_f + p_s, 1e-6) with dyadic 65536.0 quantization, and average
   switch probabilities p_sw = 0.5 * (p_sw_f + p_sw_s) quantized to 65536.0.
"""
from dataclasses import dataclass
import gc
import sys
from typing import Dict, List, Optional, Tuple

# Python 3.11.0rc1 predates CVE int_max_str_digits; shim exact signature for torch._dynamo polyfills
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = _get_int_max_str_digits
if not hasattr(sys, "set_int_max_str_digits"):
    def _set_int_max_str_digits(maxdigits: int) -> None:
        return None
    sys.set_int_max_str_digits = _set_int_max_str_digits

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from solution import ball_tracker, features_v4
from solution.common import A, D, H, X, Match


N_PLAYER_SLOTS = 11
N_PLAYER_FEATS = 10
N_CTX_BASE = 68  # 65 base features from Xtf[:, :65] + 3 ball reconstruction indicators


def _extract_ctx(Xtf_full: np.ndarray, rb: ball_tracker.ReconstructedBall) -> np.ndarray:
    b_ind = np.column_stack([
        rb.ball_conf.astype(np.float32),
        rb.vis_clean.astype(np.float32),
        np.log1p(rb.gap_len.astype(np.float32)) / 7.0,
    ])
    parts = [Xtf_full[:, :65], b_ind]
    if Xtf_full.shape[1] > 156:
        # Include compact Priority 3 base features (first 15 centroid-axis + SG ball cols)
        extra_end = min(Xtf_full.shape[1], 156 + 20)
        parts.append(Xtf_full[:, 156:extra_end])
    ctx = np.hstack(parts).astype(np.float32)
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=12.0, neginf=-12.0)
    return np.clip(ctx, -12.0, 12.0)


def _select_team_players(
    pos: np.ndarray,
    vel: np.ndarray,
    slot_idx: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bvx: np.ndarray,
    bvy: np.ndarray,
    bconf: np.ndarray,
    pitch_l: float,
    pitch_w: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select up to 11 observed players per frame and compute 10 invariant kinematics channels."""
    n = pos.shape[0]
    if len(slot_idx) == 0:
        return np.zeros((n, N_PLAYER_SLOTS, N_PLAYER_FEATS), dtype=np.float32), np.zeros(
            (n, N_PLAYER_SLOTS), dtype=np.float32
        )

    p = pos[:, slot_idx, :]
    v = vel[:, slot_idx, :]
    finite = np.isfinite(p[:, :, 0]) & np.isfinite(p[:, :, 1])

    if len(slot_idx) > N_PLAYER_SLOTS:
        d2 = np.where(
            finite,
            (p[:, :, 0] - bx[:, None]) ** 2 + (p[:, :, 1] - by[:, None]) ** 2,
            1e12,
        )
        order = np.argsort(d2, axis=1, kind="stable")[:, :N_PLAYER_SLOTS]
        rows = np.arange(n)[:, None]
        p = p[rows, order]
        v = v[rows, order]
        finite = finite[rows, order]

    m_slots = p.shape[1]
    px = np.where(finite, p[:, :, 0], 0.0)
    py = np.where(finite, p[:, :, 1], 0.0)
    vx = np.where(finite & np.isfinite(v[:, :, 0]), v[:, :, 0], 0.0)
    vy = np.where(finite & np.isfinite(v[:, :, 1]), v[:, :, 1], 0.0)

    cnt = np.maximum(finite.sum(axis=1, keepdims=True).astype(np.float32), 1.0)
    has_any = finite.any(axis=1, keepdims=True)
    cx = np.where(has_any, px.sum(axis=1, keepdims=True) / cnt, 0.5 * pitch_l)
    cy = np.where(has_any, py.sum(axis=1, keepdims=True) / cnt, 0.5 * pitch_w)

    feats = np.zeros((n, N_PLAYER_SLOTS, N_PLAYER_FEATS), dtype=np.float32)
    mask = np.zeros((n, N_PLAYER_SLOTS), dtype=np.float32)
    mask[:, :m_slots] = finite.astype(np.float32)

    bc = bconf[:, None]
    f_slice = feats[:, :m_slots, :]
    f_slice[:, :, 0] = (px - 0.5 * pitch_l) / 52.5
    f_slice[:, :, 1] = (py - 0.5 * pitch_w) / 34.0
    f_slice[:, :, 2] = (px - cx) / 30.0
    f_slice[:, :, 3] = (py - cy) / 20.0
    f_slice[:, :, 4] = vx / 12.0
    f_slice[:, :, 5] = vy / 12.0
    f_slice[:, :, 6] = ((px - bx[:, None]) / 52.5) * bc
    f_slice[:, :, 7] = ((py - by[:, None]) / 34.0) * bc
    f_slice[:, :, 8] = ((vx - bvx[:, None]) / 12.0) * bc
    f_slice[:, :, 9] = ((vy - bvy[:, None]) / 12.0) * bc
    f_slice *= finite[:, :, None].astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=6.0, neginf=-6.0)
    feats = np.clip(feats, -6.0, 6.0)
    return feats, mask


def build_view_tensors(
    m: Match,
    rb: ball_tracker.ReconstructedBall,
    vel: np.ndarray,
    Xtf_view: np.ndarray,
    swap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (own_p, own_m, opp_p, opp_m, ctx) arrays for one H/A orientation."""
    own_code = A if swap else H
    opp_code = H if swap else A
    own_idx = np.flatnonzero(m.team == own_code)
    opp_idx = np.flatnonzero(m.team == opp_code)

    bx = rb.ball_hat[:, 0].astype(np.float32)
    by = rb.ball_hat[:, 1].astype(np.float32)
    bvx = rb.ball_vel[:, 0].astype(np.float32)
    bvy = rb.ball_vel[:, 1].astype(np.float32)
    bconf = rb.ball_conf.astype(np.float32)
    pl, pw = float(m.pitch[0]), float(m.pitch[1])

    own_p, own_m = _select_team_players(m.pos, vel, own_idx, bx, by, bvx, bvy, bconf, pl, pw)
    opp_p, opp_m = _select_team_players(m.pos, vel, opp_idx, bx, by, bvx, bvy, bconf, pl, pw)
    ctx = _extract_ctx(Xtf_view, rb)
    return own_p, own_m, opp_p, opp_m, ctx


class _TCNResBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = 0.12):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, width)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(width, width, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.gelu(x + h)


class PlayerSetTCNBiGRU(nn.Module):
    """Set-pooled player encoder + multi-scale dilated TCN + BiGRU for live H/A & switch."""

    def __init__(self, n_ctx: int, width: int = 96, dropout: float = 0.12):
        super().__init__()
        self.n_ctx = n_ctx
        self.player_mlp = nn.Sequential(
            nn.Linear(N_PLAYER_FEATS, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(n_ctx, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # own(mean+max=64) + opp(mean+max=64) + diff(64) + ctx(64) = 256 -> width
        self.in_proj = nn.Sequential(
            nn.Linear(256, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            *[_TCNResBlock(width, dilation=d, dropout=dropout) for d in (1, 2, 4, 8, 16, 32)]
        )
        self.bigru = nn.GRU(width, width // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.post_norm = nn.LayerNorm(width)
        self.head_own = nn.Linear(width, 1)
        self.head_sw = nn.Linear(width, 1)

    def _pool_team(self, p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # p: (B, T, 11, 10), m: (B, T, 11)
        h = self.player_mlp(p)
        m_exp = m.unsqueeze(-1)
        h_masked = h * m_exp
        denom = torch.clamp(m_exp.sum(dim=2), min=1.0)
        mean_pool = h_masked.sum(dim=2) / denom
        # Masked max pool (safe when empty mask)
        h_for_max = h.masked_fill(m_exp < 0.5, -1e4)
        max_pool = h_for_max.max(dim=2).values
        has_any = (m_exp.sum(dim=2) > 0.5).to(h.dtype)
        max_pool = max_pool * has_any
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        own_p: torch.Tensor,
        own_m: torch.Tensor,
        opp_p: torch.Tensor,
        opp_m: torch.Tensor,
        ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_own = self._pool_team(own_p, own_m)
        e_opp = self._pool_team(opp_p, opp_m)
        e_ctx = self.ctx_proj(ctx)
        h = torch.cat([e_own, e_opp, e_own - e_opp, e_ctx], dim=-1)
        h = self.in_proj(h)  # (B, T, W)
        h_tcn = self.tcn(h.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        h_gru, _ = self.bigru(h_tcn)
        h_out = self.post_norm(h_tcn + h_gru)
        return self.head_own(h_out).squeeze(-1), self.head_sw(h_out).squeeze(-1)


@dataclass
class NeuralTeamModel:
    state_dict: dict
    ctx_mean: np.ndarray
    ctx_std: np.ndarray
    n_ctx: int
    width: int = 96
    use_p3: bool = False


def _compute_targets_and_weights(
    gt: np.ndarray,
    blocks: List[Tuple[int, int]],
    swap: bool,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute live-ownership binary target, switch target (±4f), and sample weights."""
    n = len(gt)
    own_code = A if swap else H
    opp_code = H if swap else A
    live = (gt == H) | (gt == A)

    y_own = np.full(n, -1.0, dtype=np.float32)
    y_own[gt == own_code] = 1.0
    y_own[gt == opp_code] = 0.0

    w_own = np.ones(n, dtype=np.float32)
    y_sw = np.zeros(n, dtype=np.float32)
    w_sw = np.zeros(n, dtype=np.float32)

    for b0, b1 in blocks:
        g_blk = gt[b0:b1]
        l_blk = live[b0:b1]
        # Upweight short spells <= 50 frames (2.0s)
        edges = np.flatnonzero(g_blk[1:] != g_blk[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(g_blk)]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            if l_blk[s]:
                dur = e - s
                if dur <= 25:
                    w_own[b0 + s : b0 + e] = 3.0
                elif dur <= 50:
                    w_own[b0 + s : b0 + e] = 2.5

        # Live H <-> A transitions inside this contiguous block
        sw_pts = np.flatnonzero(l_blk[:-1] & l_blk[1:] & (g_blk[:-1] != g_blk[1:])) + 1
        sw_mask = np.zeros(len(g_blk), dtype=bool)
        bnd_mask = np.zeros(len(g_blk), dtype=bool)
        for sp in sw_pts:
            sw_mask[max(0, sp - 4) : min(len(g_blk), sp + 4)] = True
            bnd_mask[max(0, sp - 6) : min(len(g_blk), sp + 6)] = True
        y_sw[b0:b1] = np.where(sw_mask & l_blk, 1.0, 0.0)
        w_own[b0:b1] = np.maximum(w_own[b0:b1], np.where(bnd_mask & l_blk, 1.8, 1.0))
        w_sw[b0:b1] = np.where(l_blk, 1.0, 0.0)

    # Extra reception boost if ELASTIC events exist
    for r in meta.get("elastic_recv_idx", []) or []:
        if 0 <= r < n and live[r]:
            s, e = max(0, r - 6), min(n, r + 6)
            w_own[s:e] = np.maximum(w_own[s:e], 2.2)

    w_own[~live] = 0.0
    return y_own, w_own, y_sw, w_sw


def _sample_windows(
    gt: np.ndarray,
    blocks: List[Tuple[int, int]],
    w_own: np.ndarray,
    rng: np.random.Generator,
    win_len: int = 1500,
    max_wins: int = 10,
) -> List[Tuple[int, int]]:
    """Select diverse contiguous live windows per match, prioritizing transitions and hard segments."""
    cands = []
    for b0, b1 in blocks:
        min_live = max(40, min(250, win_len // 3))
        if b1 - b0 < win_len:
            if ((gt[b0:b1] == H) | (gt[b0:b1] == A)).sum() >= min_live:
                cands.append((b0, b1, float(w_own[b0:b1].sum())))
            continue
        stride = max(60, min(600, win_len // 2))
        for s in range(b0, b1 - win_len + 1, stride):
            e = s + win_len
            n_live = int(((gt[s:e] == H) | (gt[s:e] == A)).sum())
            if n_live >= min_live:
                score = float(w_own[s:e].sum()) + rng.uniform(0.0, 50.0)
                cands.append((s, e, score))
    if len(cands) <= max_wins:
        return [(s, e) for s, e, _ in cands]
    # Keep mixture of highest-weight windows (short spells / transitions) and uniform spread
    cands.sort(key=lambda x: -x[2])
    top_k = max_wins // 2
    picked = [(s, e) for s, e, _ in cands[:top_k]]
    rest = cands[top_k:]
    idx = np.linspace(0, len(rest) - 1, max_wins - top_k, dtype=int)
    for ii in idx:
        picked.append((rest[ii][0], rest[ii][1]))
    return picked


def extract_match_neural_windows(
    m: Match,
    rb: ball_tracker.ReconstructedBall,
    vel: np.ndarray,
    Xtf_f: np.ndarray,
    Xtf_s: np.ndarray,
    include_dead: bool = True,
    win_len: int = 1500,
    max_wins: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> List[tuple]:
    """Extract float16 symmetric window tuples for one match cleanly from Pass-1 features."""
    if rng is None:
        rng = np.random.default_rng(0)
    own_pf, own_mf, opp_pf, opp_mf, ctx_f = build_view_tensors(m, rb, vel, Xtf_f, swap=False)
    own_ps, own_ms, opp_ps, opp_ms, ctx_s = build_view_tensors(m, rb, vel, Xtf_s, swap=True)

    y_f, w_f, ysw_f, wsw_f = _compute_targets_and_weights(m.gt, m.blocks, swap=False, meta=m.meta)
    y_s, w_s, ysw_s, wsw_s = _compute_targets_and_weights(m.gt, m.blocks, swap=True, meta=m.meta)

    n_w = max_wins if include_dead else max(3, max_wins // 2)
    wins = _sample_windows(m.gt, m.blocks, w_f, rng, win_len=win_len, max_wins=n_w)

    out = []
    for s, e in wins:
        if e - s < win_len:
            continue
        for o_p, o_m, x_p, x_m, c_v, yo, wo, ys, ws in [
            (own_pf[s:e], own_mf[s:e], opp_pf[s:e], opp_mf[s:e], ctx_f[s:e], y_f[s:e], w_f[s:e], ysw_f[s:e], wsw_f[s:e]),
            (own_ps[s:e], own_ms[s:e], opp_ps[s:e], opp_ms[s:e], ctx_s[s:e], y_s[s:e], w_s[s:e], ysw_s[s:e], wsw_s[s:e]),
        ]:
            out.append((
                o_p.astype(np.float16),
                o_m.astype(np.float16),
                x_p.astype(np.float16),
                x_m.astype(np.float16),
                c_v.astype(np.float16),
                yo.astype(np.float32),
                wo.astype(np.float32),
                ys.astype(np.float32),
                ws.astype(np.float32),
            ))
    return out


def fit_neural_team_from_windows(
    windows_raw: List[tuple],
    epochs: int = 14,
    seed: int = 0,
    use_p3: bool = False,
    use_gpu: bool = True,
) -> NeuralTeamModel:
    """Train PlayerSetTCNBiGRU from pre-extracted window tuples on GPU/CPU."""
    assert len(windows_raw) > 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    raw_ctx = int(windows_raw[0][4].shape[1])
    n_ctx = min(raw_ctx, N_CTX_BASE if not use_p3 else 88)
    sum_ctx = np.zeros(n_ctx, dtype=np.float64)
    sum_sq_ctx = np.zeros(n_ctx, dtype=np.float64)
    cnt_ctx = 0
    for w_item in windows_raw:
        c_v = w_item[4][:, :n_ctx]
        sum_ctx += c_v.sum(axis=0, dtype=np.float64)
        sum_sq_ctx += (c_v.astype(np.float64) ** 2).sum(axis=0)
        cnt_ctx += len(c_v)

    ctx_mean = (sum_ctx / cnt_ctx).astype(np.float32)
    ctx_var = np.maximum(sum_sq_ctx / cnt_ctx - ctx_mean.astype(np.float64) ** 2, 1e-4)
    ctx_std = np.sqrt(ctx_var).astype(np.float32)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = PlayerSetTCNBiGRU(n_ctx=n_ctx, width=96, dropout=0.12).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    n_win = len(windows_raw)
    batch_size = 16
    mean_t = torch.from_numpy(ctx_mean).to(device)
    std_t = torch.from_numpy(ctx_std).to(device)

    model.train()
    for ep in range(epochs):
        perm = rng.permutation(n_win)
        for i in range(0, n_win, batch_size):
            b_idx = perm[i : i + batch_size]
            op = torch.from_numpy(np.stack([windows_raw[j][0] for j in b_idx])).to(device=device, dtype=torch.float32)
            om = torch.from_numpy(np.stack([windows_raw[j][1] for j in b_idx])).to(device=device, dtype=torch.float32)
            xp = torch.from_numpy(np.stack([windows_raw[j][2] for j in b_idx])).to(device=device, dtype=torch.float32)
            xm = torch.from_numpy(np.stack([windows_raw[j][3] for j in b_idx])).to(device=device, dtype=torch.float32)
            cx = torch.from_numpy(np.stack([windows_raw[j][4][:, :n_ctx] for j in b_idx])).to(device=device, dtype=torch.float32)
            cx = torch.clamp((cx - mean_t) / std_t, -10.0, 10.0)

            yo = torch.from_numpy(np.stack([windows_raw[j][5] for j in b_idx])).to(device=device)
            wo = torch.from_numpy(np.stack([windows_raw[j][6] for j in b_idx])).to(device=device)
            ys = torch.from_numpy(np.stack([windows_raw[j][7] for j in b_idx])).to(device=device)
            ws = torch.from_numpy(np.stack([windows_raw[j][8] for j in b_idx])).to(device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                log_own, log_sw = model(op, om, xp, xm, cx)
                live_m = (wo > 0.0) & (yo >= 0.0)
                if live_m.any():
                    loss_own = F.binary_cross_entropy_with_logits(
                        log_own[live_m], yo[live_m], weight=wo[live_m], reduction="sum"
                    ) / torch.clamp(wo[live_m].sum(), min=1.0)
                else:
                    loss_own = torch.tensor(0.0, device=device)

                sw_m = ws > 0.0
                if sw_m.any():
                    pw = torch.where(ys[sw_m] > 0.5, 6.0, 1.0)
                    loss_sw = F.binary_cross_entropy_with_logits(
                        log_sw[sw_m], ys[sw_m], weight=pw, reduction="mean"
                    )
                else:
                    loss_sw = torch.tensor(0.0, device=device)

                loss = loss_own + 0.35 * loss_sw

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return NeuralTeamModel(
        state_dict=state_cpu,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        n_ctx=n_ctx,
        width=96,
        use_p3=use_p3,
    )


def fit_neural_team(
    specs: List[tuple],
    imp: Optional[ball_tracker.BallConstellationModel],
    epochs: int = 14,
    seed: int = 0,
    use_p3: bool = False,
    use_gpu: bool = True,
    win_len: int = 1500,
    max_wins_per_match: int = 8,
) -> NeuralTeamModel:
    """Train PlayerSetTCNBiGRU across lazy match builders using CUDA AMP."""
    rng = np.random.default_rng(seed)
    windows_raw = []
    for label, fold, include_dead, _, builder in specs:
        m = builder()
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
        rb = ball_tracker.reconstruct_ball(m, imputer=imp, vel=vel)
        Xtf_f = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=use_p3)
        Xtf_s = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=use_p3)
        windows_raw.extend(
            extract_match_neural_windows(
                m, rb, vel, Xtf_f, Xtf_s, include_dead=include_dead, win_len=win_len, max_wins=max_wins_per_match, rng=rng
            )
        )
        del Xtf_f, Xtf_s
    return fit_neural_team_from_windows(
        windows_raw, epochs=epochs, seed=seed, use_p3=use_p3, use_gpu=use_gpu
    )


def _predict_single_view(
    net: PlayerSetTCNBiGRU,
    device: torch.device,
    m: Match,
    rb: ball_tracker.ReconstructedBall,
    vel: np.ndarray,
    Xtf_view: np.ndarray,
    ctx_mean: np.ndarray,
    ctx_std: np.ndarray,
    swap: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    own_p, own_m, opp_p, opp_m, ctx = build_view_tensors(m, rb, vel, Xtf_view, swap=swap)
    ctx_s = ctx[:, : len(ctx_mean)]
    ctx_n = np.clip((ctx_s - ctx_mean) / ctx_std, -10.0, 10.0).astype(np.float32)

    n = m.n
    p_own = np.full(n, 0.5, dtype=np.float64)
    p_sw = np.full(n, 0.01, dtype=np.float64)

    net.bigru.flatten_parameters()
    step = 4500
    halo = 500
    with torch.no_grad():
        for b0, b1 in m.blocks:
            if b1 <= b0:
                continue
            for t0 in range(b0, b1, step):
                t1 = min(b1, t0 + step)
                s = max(b0, t0 - halo)
                e = min(b1, t1 + halo)
                op_t = torch.from_numpy(own_p[s:e][None]).to(device=device, dtype=torch.float32)
                om_t = torch.from_numpy(own_m[s:e][None]).to(device=device, dtype=torch.float32)
                xp_t = torch.from_numpy(opp_p[s:e][None]).to(device=device, dtype=torch.float32)
                xm_t = torch.from_numpy(opp_m[s:e][None]).to(device=device, dtype=torch.float32)
                cx_t = torch.from_numpy(ctx_n[s:e][None]).to(device=device, dtype=torch.float32)
                lo, ls = net(op_t, om_t, xp_t, xm_t, cx_t)
                lo_np = torch.sigmoid(lo[0]).cpu().numpy().astype(np.float64)
                ls_np = torch.sigmoid(ls[0]).cpu().numpy().astype(np.float64)
                p_own[t0:t1] = lo_np[t0 - s : t1 - s]
                p_sw[t0:t1] = ls_np[t0 - s : t1 - s]

    return p_own, p_sw


def predict_neural_team(
    bundle: NeuralTeamModel,
    m: Match,
    rb: Optional[ball_tracker.ReconstructedBall] = None,
    vel: Optional[np.ndarray] = None,
    Xtf: Optional[np.ndarray] = None,
    Xts: Optional[np.ndarray] = None,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run strictly equivariant dual-orientation prediction -> (p_ha_nn, p_sw_nn)."""
    if vel is None:
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    if rb is None:
        rb = ball_tracker.reconstruct_ball(m, vel=vel)
    if Xtf is None:
        Xtf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=bundle.use_p3)
    if Xts is None:
        Xts = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=bundle.use_p3)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    net = PlayerSetTCNBiGRU(n_ctx=bundle.n_ctx, width=bundle.width, dropout=0.0).to(device)
    net.load_state_dict(bundle.state_dict)
    net.eval()

    ph, psw_f = _predict_single_view(
        net, device, m, rb, vel, Xtf, bundle.ctx_mean, bundle.ctx_std, swap=False
    )
    pa, psw_s = _predict_single_view(
        net, device, m, rb, vel, Xts, bundle.ctx_mean, bundle.ctx_std, swap=True
    )

    d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
    d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
    p_ha_nn = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)

    psw_avg = 0.5 * (psw_f + psw_s)
    p_sw_nn = np.round(np.clip(psw_avg, 1e-4, 1.0 - 1e-4) * 65536.0) / 65536.0
    return p_ha_nn, p_sw_nn.astype(np.float32)
