"""Raw-input, exactly H<->A antisymmetric sequence model for live team ownership (NN v2).

Differences from neural_team.py (v1):
- Reads raw compacted player sets + the cleaned raw ball; all ball-gap bridging features
  (last/next seen position, linear interpolation, gap timers) are computed on the fly, so
  training can draw a fresh random ball-gap pattern for every window instead of a handful of
  precomputed degradations. Trains on every frame of every match (random 60 s crops).
- Antisymmetric by construction: logit_H = g(H as own, A as opp) - g(A as own, H as opp).
- Predicts ONLY P(H | live) and a symmetric live-switch probability. It never outputs or
  modifies P(dead), so the pipeline's ball-free dead head (dead_leak == 0) is untouched.
"""
from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Tuple

if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda maxdigits: None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from solution import ball_tracker
from solution.common import A, D, FPS, H, X, Match
from solution.neural_team import _compute_targets_and_weights

K_SLOTS = 11
K_NEAR = 3
N_PF = 10
N_NK = 8 * K_NEAR
N_TS = 7
N_CTX = 15
CAP = 1500.0


def compact_match(m: Match) -> dict:
    """Per-frame permutation-free player sets (metres, pitch-centred) + cleaned raw ball."""
    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    pl, pw = float(m.pitch[0]), float(m.pitch[1])
    n = m.n
    P = np.zeros((n, 2, K_SLOTS, 4), dtype=np.float16)
    M = np.zeros((n, 2, K_SLOTS), dtype=bool)
    rows = np.arange(n)[:, None]
    for ti, code in enumerate((H, A)):
        idx = np.flatnonzero(m.team == code)
        if len(idx) == 0:
            continue
        p = m.pos[:, idx]
        v = vel[:, idx]
        fin = np.isfinite(p[:, :, 0]) & np.isfinite(p[:, :, 1])
        order = np.argsort(~fin, axis=1, kind="stable")[:, :K_SLOTS]
        p, v, fin = p[rows, order], v[rows, order], fin[rows, order]
        k = p.shape[1]
        vfin = fin & np.isfinite(v[:, :, 0]) & np.isfinite(v[:, :, 1])
        P[:, ti, :k, 0] = np.where(fin, p[:, :, 0] - 0.5 * pl, 0.0)
        P[:, ti, :k, 1] = np.where(fin, p[:, :, 1] - 0.5 * pw, 0.0)
        P[:, ti, :k, 2] = np.where(vfin, v[:, :, 0], 0.0)
        P[:, ti, :k, 3] = np.where(vfin, v[:, :, 1], 0.0)
        M[:, ti, :k] = fin
    clean, _, _ = ball_tracker.clean_raw_ball(m.ball, m.blocks, m.pitch)
    ball = (clean - np.array([0.5 * pl, 0.5 * pw], dtype=np.float32)).astype(np.float32)
    return dict(name=m.name, P=P, M=M, ball=ball, gt=m.gt, blocks=list(m.blocks))


def featurize(P: torch.Tensor, M: torch.Tensor, ball: torch.Tensor, fps: float = FPS):
    """P (B,T,2,K,4) metres, M (B,T,2,K) float, ball (B,T,2) with NaN where missing."""
    B, T = ball.shape[:2]
    dev = ball.device
    vis = torch.isfinite(ball[..., 0])
    ar = torch.arange(T, device=dev).expand(B, T)
    last = torch.cummax(torch.where(vis, ar, torch.full_like(ar, -1)), dim=1).values
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(vis, ar, torch.full_like(ar, T)), [1]), dim=1).values, [1])
    b0 = torch.nan_to_num(ball, nan=0.0)
    has_l, has_n = last >= 0, nxt < T
    bl = torch.gather(b0, 1, last.clamp(min=0)[..., None].expand(-1, -1, 2))
    bn = torch.gather(b0, 1, nxt.clamp(max=T - 1)[..., None].expand(-1, -1, 2))
    since = torch.where(has_l, (ar - last).float(), torch.full((B, T), CAP, device=dev)).clamp(max=CAP)
    until = torch.where(has_n, (nxt - ar).float(), torch.full((B, T), CAP, device=dev)).clamp(max=CAP)
    alpha = (since / (since + until).clamp(min=1.0))[..., None]
    both = (has_l & has_n)[..., None]
    lin = torch.where(both, bl + alpha * (bn - bl), torch.where(has_l[..., None], bl, bn))
    anyb = (has_l | has_n).float()
    bl = bl * has_l[..., None]
    bn = bn * has_n[..., None]

    pad = F.pad(lin.transpose(1, 2), (2, 2), mode="replicate").transpose(1, 2)
    vb = (pad[:, 4:] - pad[:, :-4]) * (fps / 4.0)
    vb = vb.clamp(-40.0, 40.0)
    padv = F.pad(vb.transpose(1, 2), (2, 2), mode="replicate").transpose(1, 2)
    acc = ((padv[:, 4:] - padv[:, :-4]) * (fps / 4.0)).norm(dim=-1).clamp(max=200.0)
    spd_b = vb.norm(dim=-1)

    px, py, vx, vy = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    bx, by = lin[..., 0][:, :, None, None], lin[..., 1][:, :, None, None]
    dx, dy = px - bx, py - by
    dist = torch.sqrt(dx * dx + dy * dy + 1e-4)
    rvx, rvy = vx - vb[..., 0][:, :, None, None], vy - vb[..., 1][:, :, None, None]
    closing = -(dx * rvx + dy * rvy) / dist
    ab = anyb[:, :, None, None]
    pf = torch.stack([
        px / 52.5, py / 34.0, vx / 8.0, vy / 8.0,
        (dx / 15.0).clamp(-4, 4) * ab, (dy / 15.0).clamp(-4, 4) * ab,
        torch.exp(-dist / 2.0) * ab, torch.exp(-dist / 8.0) * ab,
        (closing / 8.0).clamp(-3, 3) * ab, torch.sqrt(vx * vx + vy * vy + 1e-6) / 8.0,
    ], dim=-1) * M[..., None]

    dmask = torch.where(M > 0.5, dist, torch.full_like(dist, 1e4))
    nd, ni = torch.topk(dmask, K_NEAR, dim=-1, largest=False)
    valid = (nd < 1e3).float() * ab
    g = lambda a: torch.gather(a, -1, ni)
    nk = torch.stack([
        torch.exp(-nd / 2.0), (nd / 15.0).clamp(max=4.0), (g(dx) / 15.0).clamp(-4, 4), (g(dy) / 15.0).clamp(-4, 4),
        g(vx) / 8.0, g(vy) / 8.0, (g(closing) / 8.0).clamp(-3, 3), torch.ones_like(nd),
    ], dim=-1) * valid[..., None]
    nk = nk.flatten(-2)  # (B,T,2,24)

    cnt = M.sum(-1).clamp(min=1.0)
    mean = lambda a: (a * M).sum(-1) / cnt
    cx, cy = mean(px), mean(py)
    sx = torch.sqrt(mean((px - cx[..., None]) ** 2) + 1e-4)
    sy = torch.sqrt(mean((py - cy[..., None]) ** 2) + 1e-4)
    ts = torch.stack([M.sum(-1) / 11.0, cx / 52.5, cy / 34.0, sx / 20.0, sy / 20.0, mean(vx) / 8.0, mean(vy) / 8.0], dim=-1)

    ctx = torch.stack([
        vis.float(), has_l.float(), has_n.float(),
        torch.log1p(since * FPS / fps) / 7.3, torch.log1p(until * FPS / fps) / 7.3,
        lin[..., 0] / 52.5 * anyb, lin[..., 1] / 34.0 * anyb,
        vb[..., 0] / 15.0 * anyb, vb[..., 1] / 15.0 * anyb, spd_b / 15.0 * anyb,
        bl[..., 0] / 52.5, bl[..., 1] / 34.0, bn[..., 0] / 52.5, bn[..., 1] / 34.0,
        torch.log1p(acc) / 5.0 * anyb,
    ], dim=-1)
    return pf, nk, ts, ctx


class _Res(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, width)
        self.c2 = nn.Conv1d(width, width, 1)
        self.n2 = nn.GroupNorm(8, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.n1(self.c1(x))))
        return F.gelu(x + self.n2(self.c2(h)))


class OwnNet(nn.Module):
    def __init__(self, width: int = 160, pdim: int = 48, dropout: float = 0.1, gru_layers: int = 2,
                 frame_stride: int = 1):
        super().__init__()
        self.frame_stride = frame_stride
        self.pmlp = nn.Sequential(nn.Linear(N_PF, pdim), nn.GELU(), nn.Linear(pdim, pdim), nn.GELU())
        d_in = 2 * (2 * pdim) + 2 * N_NK + 2 * N_TS + N_CTX
        self.inp = nn.Sequential(nn.Linear(d_in, width), nn.LayerNorm(width), nn.GELU())
        self.tcn = nn.Sequential(*[_Res(width, d, dropout) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.gru = nn.GRU(width, width // 2, num_layers=gru_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(width)
        self.h_own = nn.Linear(width, 1)
        self.h_sw = nn.Linear(width, 1)

    def _pool(self, pf, M):
        h = self.pmlp(pf)                               # (B,T,2,K,p)
        m = M[..., None]
        mean = (h * m).sum(-2) / m.sum(-2).clamp(min=1.0)
        mx = h.masked_fill(m < 0.5, -1e4).max(dim=-2).values * (m.sum(-2) > 0.5).float()
        return torch.cat([mean, mx], dim=-1)            # (B,T,2,2p)

    def _half(self, e, nk, ts, ctx, own: int):
        opp = 1 - own
        x = torch.cat([e[:, :, own], e[:, :, opp], nk[:, :, own], nk[:, :, opp], ts[:, :, own], ts[:, :, opp], ctx], dim=-1)
        h = self.inp(x)
        h = self.tcn(h.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        self.gru.flatten_parameters()
        hg, _ = self.gru(h)
        h = self.norm(h + hg)
        return self.h_own(h).squeeze(-1), self.h_sw(h).squeeze(-1)

    def forward(self, P, M, ball):
        pf, nk, ts, ctx = featurize(P, M, ball, fps=FPS / self.frame_stride)
        e = self._pool(pf, M)
        g0, s0 = self._half(e, nk, ts, ctx, 0)
        g1, s1 = self._half(e, nk, ts, ctx, 1)
        return g0 - g1, 0.5 * (s0 + s1)


def _bursts(rng, T, target, med, short_p=0.45):
    m = np.zeros(T, dtype=bool)
    tot, cur, tries = int(target * T), 0, 0
    while cur < tot and tries < 400:
        tries += 1
        L = int(rng.integers(8, 80)) if rng.random() < short_p else int(np.clip(rng.lognormal(np.log(med), 0.75), 80, 900))
        s = int(rng.integers(-L // 2, T))
        a, b = max(0, s), min(T, s + L)
        if b <= a:
            continue
        cur += (b - a) - int(m[a:b].sum())
        m[a:b] = True
    return m


def sample_gap_mask(rng: np.random.Generator, gt_win: np.ndarray, policy: str = "mixed") -> np.ndarray:
    """Random ball-missing mask: none / ATD-like / uniform / long bursts / live-only."""
    T = len(gt_win)
    if policy == "none":
        return np.zeros(T, dtype=bool)
    if policy == "short":
        return _bursts(rng, T, rng.uniform(0.15, 0.5), 40.0, short_p=1.0)
    if policy != "mixed":
        raise ValueError(f"Unknown gap policy: {policy}")
    dead = gt_win == D
    live = (gt_win == H) | (gt_win == A)
    r = rng.random()
    if r < 0.12:
        return np.zeros(T, dtype=bool)
    if r < 0.42:
        return (_bursts(rng, T, rng.uniform(0.08, 0.30), 90.0) & live) | (_bursts(rng, T, rng.uniform(0.5, 0.95), 280.0) & dead)
    if r < 0.67:
        miss = _bursts(rng, T, rng.uniform(0.25, 0.60), 220.0)
    elif r < 0.87:
        miss = np.zeros(T, dtype=bool)
        cur = int(rng.integers(0, 600))
        while cur < T:
            w = int(rng.integers(125, 751))
            miss[cur : cur + w] = True
            cur += w + int(rng.integers(200, 601))
        miss |= _bursts(rng, T, rng.uniform(0.0, 0.10), 40.0, short_p=1.0)
    else:
        return _bursts(rng, T, rng.uniform(0.2, 0.5), 140.0) & live
    if rng.random() < 0.5:
        miss |= _bursts(rng, T, 0.8, 280.0) & dead
    return miss


@dataclass
class NeuralV2Model:
    state_dict: dict
    width: int = 160
    pdim: int = 48
    gru_layers: int = 2
    frame_stride: int = 1


def prepare_record(c: dict, weight: float = 1.0, alt: Optional[dict] = None, meta: Optional[dict] = None) -> dict:
    """Attach loss targets/weights to a compact_match dict. alt = degraded-player variant."""
    y, w, ysw, wsw = _compute_targets_and_weights(c["gt"], c["blocks"], swap=False, meta=meta or {})
    return dict(c, y=y, w=w, ysw=ysw, wsw=wsw, weight=weight, alt=alt)


def fit(records: List[dict], steps: int = 3000, batch: int = 24, T: int = 1500, seed: int = 0,
        width: int = 160, pdim: int = 48, gru_layers: int = 2, lr: float = 2e-3, log_every: int = 0,
        frame_stride: int = 1, gap_policy: str = "mixed") -> NeuralV2Model:
    if not records or steps < 1 or batch < 1 or frame_stride < 1:
        raise ValueError("Nonempty records, positive steps/batch/stride are required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OwnNet(width=width, pdim=pdim, gru_layers=gru_layers, frame_stride=frame_stride).to(dev)
    ema = OwnNet(width=width, pdim=pdim, gru_layers=gru_layers, frame_stride=frame_stride).to(dev)
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    # (record, block) sampling table weighted by record weight x live frames in block
    table, probs = [], []
    for ri, r in enumerate(records):
        for b0, b1 in r["blocks"]:
            if b1 - b0 >= T + 2:
                table.append((ri, b0, b1))
                probs.append(r["weight"] * (b1 - b0))
    probs = np.asarray(probs, dtype=np.float64)
    if len(table) == 0 or probs.sum() <= 0:
        raise ValueError("No training blocks long enough for the requested window")
    probs /= probs.sum()

    net.train()
    run = 0.0
    for step in range(steps):
        Ps, Ms, Bs, Ys, Ws, Ysw, Wsw = [], [], [], [], [], [], []
        for ti in rng.choice(len(table), size=batch, p=probs):
            ri, b0, b1 = table[ti]
            r = records[ri]
            s = int(rng.integers(b0, b1 - T))
            src = r["alt"] if (r["alt"] is not None and rng.random() < 0.5) else r
            P = src["P"][s : s + T].astype(np.float32)
            M = src["M"][s : s + T].astype(np.float32)
            ball = r["ball"][s : s + T].copy()
            if rng.random() < 0.7:
                from scipy.ndimage import gaussian_filter1d
                d = gaussian_filter1d(rng.normal(0, 1, size=(T, 2)).astype(np.float32), sigma=12.0, axis=0)
                ball += d * (0.25 / max(float(d.std()), 1e-6)) + rng.normal(0, 0.03, size=(T, 2)).astype(np.float32)
            ball[sample_gap_mask(rng, r["gt"][s : s + T], policy=gap_policy)] = np.nan
            if rng.random() < 0.5:
                P[..., 0] *= -1; P[..., 2] *= -1; ball[:, 0] *= -1
            if rng.random() < 0.5:
                P[..., 1] *= -1; P[..., 3] *= -1; ball[:, 1] *= -1
            Ps.append(P[::frame_stride]); Ms.append(M[::frame_stride]); Bs.append(ball[::frame_stride])
            Ys.append(r["y"][s : s + T : frame_stride]); Ws.append(r["w"][s : s + T : frame_stride])
            Ysw.append(r["ysw"][s : s + T : frame_stride]); Wsw.append(r["wsw"][s : s + T : frame_stride])
        tt = lambda a: torch.from_numpy(np.stack(a)).to(dev)
        P, M, ball, y, w, ysw, wsw = tt(Ps), tt(Ms), tt(Bs), tt(Ys), tt(Ws), tt(Ysw), tt(Wsw)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            lo, ls = net(P, M, ball)
        lo, ls = lo.float(), ls.float()
        lm = (w > 0) & (y >= 0)
        loss = (F.binary_cross_entropy_with_logits(lo[lm], y[lm], weight=w[lm], reduction="sum") / w[lm].sum().clamp(min=1.0))
        sm = wsw > 0
        if sm.any():
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(
                ls[sm], ysw[sm], weight=torch.where(ysw[sm] > 0.5, 6.0, 1.0), reduction="mean")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            decay = min(0.999, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(decay).add_(pn.detach(), alpha=1 - decay)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{steps} loss {run:.4f}", flush=True)

    state = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
    del net, ema, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return NeuralV2Model(state_dict=state, width=width, pdim=pdim, gru_layers=gru_layers, frame_stride=frame_stride)


def predict_compact(models: List[NeuralV2Model], c: dict, chunk: int = 6000, halo: int = 750) -> Tuple[np.ndarray, np.ndarray]:
    """Average logits over models x 4 mirror flips -> (p_H_given_live, p_switch), dyadically quantized."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(c["ball"])
    lo_sum = np.zeros(n, dtype=np.float64)
    sw_sum = np.zeros(n, dtype=np.float64)
    nets = []
    if not models:
        raise ValueError("At least one trained ownership model is required")
    strides = {getattr(mb, "frame_stride", 1) for mb in models}
    if len(strides) != 1:
        raise ValueError("All ensemble models must use the same frame stride")
    frame_stride = strides.pop()
    for mb in models:
        net = OwnNet(width=mb.width, pdim=mb.pdim, gru_layers=mb.gru_layers, frame_stride=frame_stride).to(dev)
        net.load_state_dict(mb.state_dict)
        nets.append(net.eval())
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    with torch.no_grad():
        for b0, b1 in c["blocks"]:
            for t0 in range(b0, b1, chunk):
                t1 = min(b1, t0 + chunk)
                s, e = max(b0, t0 - halo), min(b1, t1 + halo)
                # Keep a block-anchored sample grid across chunk boundaries.
                s = max(b0, s - (s - b0) % frame_stride)
                P = torch.from_numpy(c["P"][s:e:frame_stride].astype(np.float32))[None].to(dev)
                M = torch.from_numpy(c["M"][s:e:frame_stride].astype(np.float32))[None].to(dev)
                ball = torch.from_numpy(c["ball"][s:e:frame_stride])[None].to(dev)
                for fx, fy in flips:
                    sgn = torch.tensor([fx, fy, fx, fy], device=dev, dtype=torch.float32)
                    Pf, bf = P * sgn, ball * sgn[:2]
                    # Explicit swap pairing: bit-exact H<->A antisymmetry regardless of kernel ordering.
                    Ps, Ms = torch.flip(Pf, [2]), torch.flip(M, [2])
                    for net in nets:
                        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                            lo_a, ls_a = net(Pf, M, bf)
                            lo_b, ls_b = net(Ps, Ms, bf)
                        xp = np.arange(s, e, frame_stride)
                        out_x = np.arange(t0, t1)
                        interp = lambda a: np.interp(out_x, xp, a[0].float().cpu().numpy().astype(np.float64))
                        la, lb = interp(lo_a), interp(lo_b)
                        sa, sb = interp(torch.sigmoid(ls_a.float())), interp(torch.sigmoid(ls_b.float()))
                        lo_sum[t0:t1] += 0.5 * (la - lb)
                        sw_sum[t0:t1] += 0.5 * (sa + sb)
    k = len(nets) * len(flips)
    d = np.tanh(0.5 * lo_sum / k)                      # = 2*sigmoid(logit) - 1
    d_q = np.round(np.clip(d, -0.9998, 0.9998) * 65536.0) / 65536.0
    p_ha = np.clip(0.5 + 0.5 * d_q, 1e-4, 1.0 - 1e-4).astype(np.float32)
    p_sw = (np.round(np.clip(sw_sum / k, 1e-4, 1.0 - 1e-4) * 65536.0) / 65536.0).astype(np.float32)
    return p_ha, p_sw


def predict_match(models: List[NeuralV2Model], m: Match) -> Tuple[np.ndarray, np.ndarray]:
    return predict_compact(models, compact_match(m))
