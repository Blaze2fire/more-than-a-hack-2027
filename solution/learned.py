"""Option B: per-frame H/A/D probabilities from models trained on EXTERNAL games only.

Two models, deliberately separated (audit finding, RESEARCH_LOG E006):
  dead model   P(dead ball) from PLAYER KINEMATICS ONLY. It never sees the ball. The external
               games have no ball during stoppages, so a model allowed to look at the ball there
               would learn whatever stoppage picture our simulation invented.
  team model   P(home | alive) from ball-and-player features, trained on LIVE frames only, where
               the ball track is real (only thinned and jittered).
They are combined as P(D), (1-P(D))*P(H|alive), (1-P(D))*P(A|alive) and decoded with the same
Viterbi decoder as Option A. Features are identity-free (slots are interchangeable).

Training data: Metrica sample games (see docs/DATA_CONTRACT.md). Never the competition match.
"""
import os
import pickle

import numpy as np

from . import model
from .common import A, D, FPS, H, HERE, X

WINDOWS = (12, 25, 50, 100)        # frames, used both backwards and forwards
CAP = 250                          # frames: cap on "time since / until" features
R_CTRL = 2.0                       # m: a player this close and closest of all "controls" the ball
MODEL_PATH = os.path.join(HERE, "configs", "gbm.pkl")
DECODE_DEFAULTS = dict(w=1.0, c_team=6.0, c_dead=15.0, c_live=10.0, min_spell=0)


def _win_mean(x, ok, w, forward):
    """Mean of x over the w frames before (or after) each frame, counting only frames where ok."""
    xs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    ks = np.concatenate(([0.0], np.cumsum(ok.astype(np.float64))))
    i = np.arange(len(x))
    lo, hi = (i + 1, np.minimum(len(x), i + 1 + w)) if forward else (np.maximum(0, i - w), i)
    hi = np.maximum(hi, lo)
    lo = np.minimum(lo, len(x))
    cnt = ks[hi] - ks[lo]
    return np.where(cnt > 0, (xs[hi] - xs[lo]) / np.maximum(cnt, 1), np.nan), cnt / float(w)


def _since(flag):
    """Frames since flag was last True (CAP if never / too long)."""
    idx = np.where(flag, np.arange(len(flag)), -10 ** 9)
    np.maximum.accumulate(idx, out=idx)
    return np.minimum(np.arange(len(flag)) - idx, CAP).astype(np.float32)


def _until(flag):
    return _since(flag[::-1])[::-1]


def _last_value(x, ok):
    """x at the most recent frame where ok (NaN before the first)."""
    idx = np.where(ok, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def _next_value(x, ok):
    return _last_value(x[::-1], ok[::-1])[::-1]


def build(m, swap=False):
    """(features (n, K) float32, names). swap=True builds the mirror-image problem (teams swapped)."""
    n = m.n
    team = (1 - m.team) if swap else m.team
    inb = np.zeros(n, dtype=bool)
    for b0, b1 in m.blocks:
        inb[b0:b1] = True
    L, W = m.pitch
    bx = np.where(inb, m.ball[:, 0], np.nan)
    by = np.where(inb, m.ball[:, 1], np.nan)
    vis = ~np.isnan(bx)

    px, py = m.pos[:, :, 0], m.pos[:, :, 1]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    us, them = team == 0, team == 1
    d_us = np.minimum(d[:, us].min(axis=1), 40.0)
    d_them = np.minimum(d[:, them].min(axis=1), 40.0)
    # second-nearest per side: how contested the ball is
    d2_us = np.minimum(np.partition(d[:, us], 1, axis=1)[:, 1], 40.0)
    d2_them = np.minimum(np.partition(d[:, them], 1, axis=1)[:, 1], 40.0)
    margin = np.where(vis, d_them - d_us, np.nan)
    near = np.where(vis, np.minimum(d_us, d_them), np.nan)
    ctrl_us = vis & (d_us < R_CTRL) & (d_us < d_them)
    ctrl_them = vis & (d_them < R_CTRL) & (d_them < d_us)

    lag = 3
    vb = np.full(n, np.nan)
    step_x, step_y = bx[2 * lag:] - bx[:-2 * lag], by[2 * lag:] - by[:-2 * lag]
    vb[lag:-lag] = np.hypot(step_x, step_y) * (FPS / (2.0 * lag))

    with np.errstate(invalid="ignore", divide="ignore"):
        def tmean(arr, mask):
            sub = arr[:, mask]
            c = (~np.isnan(sub)).sum(axis=1)
            return np.where(c > 0, np.nansum(sub, axis=1) / np.maximum(c, 1), np.nan)

        def tstd(arr, mask, mu):
            sub = arr[:, mask]
            c = (~np.isnan(sub)).sum(axis=1)
            return np.sqrt(np.where(c > 1, np.nansum((sub - mu[:, None]) ** 2, axis=1) / np.maximum(c, 1), np.nan))

        cx_us, cy_us = tmean(px, us), tmean(py, us)
        cx_them, cy_them = tmean(px, them), tmean(py, them)
        sx_us, sx_them = tstd(px, us, cx_us), tstd(px, them, cx_them)
        sy_us, sy_them = tstd(py, us, cy_us), tstd(py, them, cy_them)
        sp_us, sp_them = tmean(m.spd, us), tmean(m.spd, them)
        slow = np.where(np.isnan(m.spd), np.nan, (m.spd < 1.5).astype(np.float32))
        _c = (~np.isnan(slow)).sum(axis=1)
        slow_frac = np.where(_c > 0, np.nansum(slow, axis=1) / np.maximum(_c, 1), np.nan)
        n_tracked = (~np.isnan(px)).sum(axis=1).astype(np.float32)

    # attack direction of "us" per block: we attack away from the side our team sits on
    direction = np.ones(n)
    for b0, b1 in m.blocks:
        diff = np.nanmean(cx_them[b0:b1] - cx_us[b0:b1])
        direction[b0:b1] = 1.0 if (np.isnan(diff) or diff >= 0) else -1.0

    def vel(c):
        v = np.full(n, np.nan)
        v[12:-12] = (c[24:] - c[:-24]) * (FPS / 24.0)
        return v

    feats, names = [], []

    def add(name, arr):
        feats.append(np.asarray(arr, dtype=np.float32))
        names.append(name)

    add("vis", vis)
    add("margin", np.clip(margin, -20, 20))
    add("near", near)
    add("d2_gap_us", np.where(vis, d2_us - d_us, np.nan))
    add("d2_gap_them", np.where(vis, d2_them - d_them, np.nan))
    add("vb", vb)
    add("ball_x_dir", direction * (bx - L / 2))
    add("ball_edge_y", np.minimum(by, W - by))
    add("ball_edge_x", np.minimum(bx, L - bx))
    add("ball_rel_us", direction * (bx - cx_us))
    add("ball_rel_them", direction * (bx - cx_them))
    add("gap_centroids", direction * (cx_them - cx_us))
    add("v_us_dir", direction * vel(cx_us))
    add("v_them_dir", direction * vel(cx_them))
    add("spread_x_diff", sx_us - sx_them)
    add("spread_y_diff", sy_us - sy_them)
    add("spread_x_sum", sx_us + sx_them)
    add("spread_y_sum", sy_us + sy_them)
    add("speed_us", model._smooth_nan(sp_us, 12))
    add("speed_them", model._smooth_nan(sp_them, 12))
    add("speed_all_2s", model._smooth_nan(0.5 * (sp_us + sp_them), 50))
    add("speed_all_6s", model._smooth_nan(0.5 * (sp_us + sp_them), 150))
    add("slow_frac", model._smooth_nan(slow_frac, 25))
    add("n_tracked", model._smooth_nan(n_tracked, 25))

    ok_margin = vis
    for w in WINDOWS:
        for fwd, tag in ((False, "p"), (True, "f")):
            mm, share = _win_mean(np.nan_to_num(margin), ok_margin, w, fwd)
            add("margin_%s%d" % (tag, w), np.clip(mm, -20, 20))
            add("vis_%s%d" % (tag, w), share)
            add("ctrl_us_%s%d" % (tag, w), _win_mean(ctrl_us.astype(float), np.ones(n, bool), w, fwd)[0])
            add("ctrl_them_%s%d" % (tag, w), _win_mean(ctrl_them.astype(float), np.ones(n, bool), w, fwd)[0])
            add("vb_%s%d" % (tag, w), _win_mean(np.nan_to_num(vb), ~np.isnan(vb), w, fwd)[0])

    add("since_vis", _since(vis))
    add("until_vis", _until(vis))
    add("since_ctrl_us", _since(ctrl_us))
    add("since_ctrl_them", _since(ctrl_them))
    add("until_ctrl_us", _until(ctrl_us))
    add("until_ctrl_them", _until(ctrl_them))
    add("margin_last", np.clip(_last_value(np.nan_to_num(margin), vis), -20, 20))
    add("margin_next", np.clip(_next_value(np.nan_to_num(margin), vis), -20, 20))
    return np.stack(feats, axis=1), names


def build_dead(m):
    """Player-only, team-symmetric features for dead-vs-alive. No ball anywhere in here."""
    n = m.n
    us, them = m.team == 0, m.team == 1
    px, py = m.pos[:, :, 0], m.pos[:, :, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        def tmean(arr, mask=None):
            sub = arr if mask is None else arr[:, mask]
            c = (~np.isnan(sub)).sum(axis=1)
            return np.where(c > 0, np.nansum(sub, axis=1) / np.maximum(c, 1), np.nan)

        sp = tmean(m.spd)
        sp_us, sp_them = tmean(m.spd, us), tmean(m.spd, them)
        c = (~np.isnan(m.spd)).sum(axis=1)
        sp_sd = np.sqrt(np.where(c > 1, np.nansum((m.spd - sp[:, None]) ** 2, axis=1) / np.maximum(c, 1), np.nan))
        sp_max = np.where(c > 0, np.nanmax(np.where(np.isnan(m.spd), -1.0, m.spd), axis=1), np.nan)
        slow = tmean(np.where(np.isnan(m.spd), np.nan, (m.spd < 1.5).astype(np.float32)))
        fast = tmean(np.where(np.isnan(m.spd), np.nan, (m.spd > 4.0).astype(np.float32)))
        cx, cy = tmean(px), tmean(py)
        sx = np.sqrt(tmean((px - cx[:, None]) ** 2))
        sy = np.sqrt(tmean((py - cy[:, None]) ** 2))
        gap = np.abs(tmean(px, us) - tmean(px, them))

    def vel(v):
        out = np.full(n, np.nan)
        out[12:-12] = np.abs(v[24:] - v[:-24]) * (FPS / 24.0)
        return out

    feats, names = [], []

    def add(name, arr):
        feats.append(np.asarray(arr, dtype=np.float32))
        names.append(name)

    ok = ~np.isnan(sp)
    for half in (12, 25, 50, 150, 300):
        add("speed_c%d" % half, model._smooth_nan(sp, half))
    for w in (50, 150, 300):
        add("speed_p%d" % w, _win_mean(np.nan_to_num(sp), ok, w, False)[0])
        add("speed_f%d" % w, _win_mean(np.nan_to_num(sp), ok, w, True)[0])
    add("speed_team_gap", model._smooth_nan(np.abs(sp_us - sp_them), 25))
    add("speed_sd", model._smooth_nan(sp_sd, 25))
    add("speed_max", model._smooth_nan(sp_max, 25))
    for half in (25, 100):
        add("slow_frac_%d" % half, model._smooth_nan(slow, half))
        add("fast_frac_%d" % half, model._smooth_nan(fast, half))
    add("spread_x", model._smooth_nan(sx, 12))
    add("spread_y", model._smooth_nan(sy, 12))
    add("centroid_gap", model._smooth_nan(gap, 12))
    add("centroid_vx", vel(cx))
    add("centroid_vy", vel(cy))
    add("spread_vx", vel(sx))
    return np.stack(feats, axis=1), names


def fit(matches, stride=4, seed=0, **kw):
    """Train (dead model, team model) on external Matches with gt."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, random_state=seed)
    params.update(kw)
    Xd, yd, Xt, yt = [], [], [], []
    for m in matches:
        graded = np.flatnonzero(m.gt != X)[::stride]
        Xd.append(build_dead(m)[0][graded])
        yd.append((m.gt[graded] == D).astype(np.int8))
        alive = graded[m.gt[graded] != D]
        for swap in (False, True):
            Xt.append(build(m, swap)[0][alive])
            y = (m.gt[alive] == H)
            yt.append((~y if swap else y).astype(np.int8))
    dead = HistGradientBoostingClassifier(**params).fit(np.concatenate(Xd), np.concatenate(yd))
    team = HistGradientBoostingClassifier(**params).fit(np.concatenate(Xt), np.concatenate(yt))
    return dead, team


def log_proba(clfs, m):
    """(n, 3) log-probabilities for H, A, D."""
    dead, team = clfs
    p_dead = dead.predict_proba(build_dead(m)[0])[:, 1]
    p_home = 0.5 * (team.predict_proba(build(m, False)[0])[:, 1]
                    + 1.0 - team.predict_proba(build(m, True)[0])[:, 1])
    p = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
    return np.log(np.clip(p, 1e-6, 1.0))


def decode(logp, blocks, cfg=None):
    cfg = dict(DECODE_DEFAULTS, **(cfg or {}))
    out = np.full(len(logp), X, dtype=np.int8)
    s = cfg["w"] * logp
    for b0, b1 in blocks:
        path = model.viterbi3(s[b0:b1, 0], s[b0:b1, 1], s[b0:b1, 2],
                              cfg["c_team"], cfg["c_dead"], cfg["c_live"])
        out[b0:b1] = model.absorb_short(path, int(cfg["min_spell"]))
    return out


def save(clf, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(clf, handle)


def load(path=MODEL_PATH):
    with open(path, "rb") as handle:
        return pickle.load(handle)
