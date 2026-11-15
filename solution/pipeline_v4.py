"""End-to-end multi-stage pipeline: Ball Reconstruction -> GBDT Ensemble -> Temporal Stacker -> Viterbi4."""
from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from solution import ball_tracker, features_v4, temporal, viterbi4
from solution.ball_tracker import BallConstellationModel, ReconstructedBall
from solution.common import A, D, H, X, Match


def boundary_weights(m: Match, switch_half: int = 30, elastic_half: int = 18) -> np.ndarray:
    """Compute per-frame sample weights emphasizing state switches and ELASTIC receptions."""
    n = m.n
    w = np.ones(n, dtype=np.float32)
    gt = m.gt
    if gt is None:
        return w

    for b0, b1 in m.blocks:
        g = gt[b0:b1]
        sw = np.where((g[1:] != g[:-1]) & (g[1:] != X) & (g[:-1] != X))[0] + 1
        for idx in sw:
            lo = max(0, idx - switch_half)
            hi = min(b1 - b0, idx + switch_half + 1)
            w[b0 + lo : b0 + hi] = np.maximum(w[b0 + lo : b0 + hi], 2.5)
            lo_tight = max(0, idx - 10)
            hi_tight = min(b1 - b0, idx + 11)
            w[b0 + lo_tight : b0 + hi_tight] = np.maximum(w[b0 + lo_tight : b0 + hi_tight], 3.5)

    for key in ("elastic_recv_idx", "elastic_kick_idx"):
        evs = m.meta.get(key)
        if evs:
            boost = 3.5 if key == "elastic_recv_idx" else 2.2
            for t in evs:
                lo = max(0, int(t) - elastic_half)
                hi = min(n, int(t) + elastic_half + 1)
                w[lo:hi] = np.maximum(w[lo:hi], boost)
    return w


class EnsembleBinaryClf:
    """Weighted blend of LightGBM + XGBoost (CUDA if available) + CatBoost."""

    def __init__(self, mode: str = "lgbm", seed: int = 0, use_gpu: bool = True, weights: Optional[Tuple[float, ...]] = None):
        self.mode = mode
        self.seed = seed
        self.use_gpu = use_gpu
        self.weights = weights
        self.models: List[Tuple[str, object]] = []

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "EnsembleBinaryClf":
        import lightgbm as lgb
        import xgboost as xgb
        import catboost as cb

        self.models = []
        if self.mode == "hgb":
            m = HistGradientBoostingClassifier(
                max_iter=250, max_depth=7, learning_rate=0.06, min_samples_leaf=80,
                l2_regularization=1.0, random_state=self.seed,
            )
            m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("hgb", m))
        elif self.mode == "lgbm":
            m = lgb.LGBMClassifier(
                n_estimators=400,
                num_leaves=63,
                learning_rate=0.05,
                subsample_for_bin=100000,
                colsample_bytree=0.75,
                subsample=0.85,
                subsample_freq=1,
                min_child_samples=60,
                reg_alpha=0.5,
                reg_lambda=2.0,
                n_jobs=12,
                random_state=self.seed,
                verbose=-1,
            )
            m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("lgbm", m))
        elif self.mode == "ens3":
            lgbm_m = lgb.LGBMClassifier(
                n_estimators=450,
                num_leaves=63,
                learning_rate=0.045,
                colsample_bytree=0.75,
                subsample=0.85,
                subsample_freq=1,
                min_child_samples=60,
                reg_alpha=0.5,
                reg_lambda=2.0,
                n_jobs=12,
                random_state=self.seed,
                verbose=-1,
            )
            lgbm_m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("lgbm", lgbm_m))

            xgb_kw = dict(
                n_estimators=450,
                max_depth=6,
                learning_rate=0.045,
                colsample_bytree=0.75,
                subsample=0.85,
                min_child_weight=20.0,
                reg_alpha=0.5,
                reg_lambda=2.0,
                tree_method="hist",
                device="cuda" if self.use_gpu else "cpu",
                random_state=self.seed,
                eval_metric="logloss",
            )
            xgb_m = xgb.XGBClassifier(**xgb_kw)
            xgb_m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("xgb", xgb_m))

            cb_m = cb.CatBoostClassifier(
                iterations=700,
                depth=6,
                learning_rate=0.06,
                l2_leaf_reg=5.0,
                task_type="GPU" if self.use_gpu else "CPU",
                random_seed=self.seed,
                verbose=0,
            )
            cb_m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("cb", cb_m))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self

    def predict_proba(self, X_in: np.ndarray) -> np.ndarray:
        if len(self.models) == 1:
            return self.models[0][1].predict_proba(X_in).astype(np.float32)
        ws = self.weights or tuple([1.0 / len(self.models)] * len(self.models))
        out = None
        for w, (_, mdl) in zip(ws, self.models):
            p = mdl.predict_proba(X_in).astype(np.float64)
            out = p * w if out is None else out + p * w
        return np.clip(out, 1e-5, 1.0 - 1e-5).astype(np.float32)


def fit_temporal_stacker_from_oof(
    oof_items: List[tuple],
    stride: int = 3,
    seed: int = 0,
    mode: str = "ens3",
    use_gpu: bool = True,
) -> temporal.TemporalStacker:
    """Fit Stage-2 dead and team sequence stackers on leak-free OOF Stage-1 probabilities.

    oof_items: list of (p_dead_s1, p_ha_live_s1, Xd_base, Xf_base, match)
               or (p_dead_s1, p_ha_live_s1, Xd_base, Xf_base, Xs_base, match).
    """
    Zds, yds, wds = [], [], []
    Zts, yts, wts = [], [], []

    for item in oof_items:
        if len(item) == 6:
            p_d1, p_ha1, Xd_base, Xf_base, Xs_base, m = item
        else:
            p_d1, p_ha1, Xd_base, Xf_base, m = item
            Xs_base = None
        gt = m.gt
        assert gt is not None
        mask = np.zeros(m.n, dtype=bool)
        for b0, b1 in m.blocks:
            mask[b0:b1:stride] = True
        bw = boundary_weights(m)

        # Dead stacker only trains on matches with real dead-ball labels!
        if m.meta.get("has_real_dead_ball", True):
            dm = mask & (gt != X)
            if dm.any():
                Zd = temporal.build_dead_seq_features(p_d1, Xd_base, m.blocks)
                Zds.append(Zd[dm])
                yds.append((gt[dm] == D).astype(np.int8))
                wds.append(bw[dm])

        lm = mask & ((gt == H) | (gt == A))
        if lm.any():
            Zt_f = temporal.build_team_seq_features(p_ha1, p_d1, Xf_base, m.blocks, swap=False)
            Zt_s = temporal.build_team_seq_features(p_ha1, p_d1, Xf_base, m.blocks, swap=True, Xs_base=Xs_base)
            Zts.append(Zt_f[lm])
            yts.append((gt[lm] == H).astype(np.int8))
            wts.append(bw[lm])
            Zts.append(Zt_s[lm])
            yts.append((gt[lm] == A).astype(np.int8))
            wts.append(bw[lm])

    dead_s2 = EnsembleBinaryClf(mode=mode, seed=seed, use_gpu=use_gpu).fit(
        np.vstack(Zds), np.concatenate(yds), sample_weight=np.concatenate(wds)
    )
    team_s2 = EnsembleBinaryClf(mode=mode, seed=seed + 1, use_gpu=use_gpu).fit(
        np.vstack(Zts), np.concatenate(yts), sample_weight=np.concatenate(wts)
    )
    return temporal.TemporalStacker(dead_s2=dead_s2, team_s2=team_s2)


@dataclass
class PipelineV4:
    imputer: Optional[BallConstellationModel]
    dead_s1: EnsembleBinaryClf
    team_s1: EnsembleBinaryClf
    stacker: Optional[temporal.TemporalStacker] = None
    decode_cfg: dict = field(default_factory=lambda: dict(viterbi4.DECODE4_DEFAULTS))

    def predict_stage1(self, m: Match) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
        rb = ball_tracker.reconstruct_ball(m, imputer=self.imputer, vel=vel)
        Xd = features_v4.build_dead_features(m, vel=vel)
        Xt_f = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False)
        Xt_s = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True)
        p_d1 = np.clip(self.dead_s1.predict_proba(Xd)[:, 1], 1e-4, 1.0 - 1e-4)
        ph = self.team_s1.predict_proba(Xt_f)[:, 1].astype(np.float64)
        pa = self.team_s1.predict_proba(Xt_s)[:, 1].astype(np.float64)
        d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
        # Dyadic quantization eliminates 1-ULP asymmetry between 0.5+0.5*d and 0.5-0.5*d
        d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
        p_ha1 = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)
        return (
            p_d1,
            p_ha1,
            Xd[:, :40].astype(np.float32, copy=False),
            Xt_f[:, :65].astype(np.float32, copy=False),
            Xt_s[:, :65].astype(np.float32, copy=False),
        )

    def predict_log_proba(self, m: Match) -> np.ndarray:
        p_d1, p_ha1, Xd_base, Xf_base, Xs_base = self.predict_stage1(m)
        if self.stacker is not None:
            return self.stacker.predict_log_proba(p_d1, p_ha1, Xd_base, Xf_base, m.blocks, Xs_base=Xs_base)
        live = 1.0 - p_d1.astype(np.float64)
        d_ha = 2.0 * p_ha1.astype(np.float64) - 1.0
        ph_live = np.clip(0.5 + 0.5 * d_ha, 1e-4, 1.0 - 1e-4)
        pa_live = np.clip(0.5 - 0.5 * d_ha, 1e-4, 1.0 - 1e-4)
        p3 = np.column_stack([live * ph_live, live * pa_live, p_d1])
        return np.log(np.clip(p3, 1e-5, 1.0)).astype(np.float32)

    def predict(self, m: Match) -> np.ndarray:
        return viterbi4.decode4(self.predict_log_proba(m), m.blocks, self.decode_cfg)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PipelineV4":
        with open(path, "rb") as f:
            return pickle.load(f)

