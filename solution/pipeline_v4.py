"""End-to-end multi-stage pipeline: Ball Reconstruction -> GBDT Ensemble -> Temporal Stacker -> Viterbi4."""
from dataclasses import dataclass, field
import os
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
        threads = max(1, int(os.environ.get("MTAH_THREADS", "4")))
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
                n_jobs=threads,
                random_state=self.seed,
                verbose=-1,
            )
            m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("lgbm", m))
        elif self.mode == "xgb":
            xgb_m = xgb.XGBClassifier(
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
                n_jobs=threads,
            )
            xgb_m.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.models.append(("xgb", xgb_m))
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
                n_jobs=threads,
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
                n_jobs=threads,
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
                thread_count=threads,
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
        s2_w = 1.0
        if len(item) == 7:
            p_d1, p_ha1, Xd_base, Xf_base, Xs_base, m, s2_w = item
        elif len(item) == 6:
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
            wt_lm = bw[lm] * float(s2_w)
            Zt_f = temporal.build_team_seq_features(p_ha1, p_d1, Xf_base, m.blocks, swap=False)
            Zt_s = temporal.build_team_seq_features(p_ha1, p_d1, Xf_base, m.blocks, swap=True, Xs_base=Xs_base)
            Zts.append(Zt_f[lm])
            yts.append((gt[lm] == H).astype(np.int8))
            wts.append(wt_lm)
            Zts.append(Zt_s[lm])
            yts.append((gt[lm] == A).astype(np.int8))
            wts.append(wt_lm)

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
    team_bf_s1: Optional[EnsembleBinaryClf] = None
    bf_cols: Optional[np.ndarray] = None
    neural_model: Optional[object] = None
    neural_weight: float = 0.0
    use_p3: bool = False
    neural_v2: Optional[list] = None  # list of neural_v2.NeuralV2Model; takes precedence over neural_model
    boundary_model: Optional[object] = None
    neural_switch_mode: str = "blend"
    freeze_dead_mask: bool = False
    dead_reference_cfg: Optional[dict] = None
    line_snap_cfg: Optional[dict] = None  # None = off; dict = solution.line_snap parameters

    def _compute_s1(self, m: Match):
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
        rb = ball_tracker.reconstruct_ball(m, imputer=self.imputer, vel=vel)
        use_p3 = bool(getattr(self, "use_p3", False))
        Xd = features_v4.build_dead_features(m, vel=vel)
        Xt_f = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=use_p3)
        Xt_s = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=use_p3)
        p_d1 = np.clip(self.dead_s1.predict_proba(Xd)[:, 1], 1e-4, 1.0 - 1e-4)
        ph = self.team_s1.predict_proba(Xt_f)[:, 1].astype(np.float64)
        pa = self.team_s1.predict_proba(Xt_s)[:, 1].astype(np.float64)
        d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
        # Dyadic quantization eliminates 1-ULP asymmetry between 0.5+0.5*d and 0.5-0.5*d
        d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
        p_ha1 = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)

        p_ha_bf = None
        team_bf_s1 = getattr(self, "team_bf_s1", None)
        bf_cols = getattr(self, "bf_cols", None)
        if team_bf_s1 is not None and bf_cols is not None:
            ph_b = team_bf_s1.predict_proba(Xt_f[:, bf_cols])[:, 1].astype(np.float64)
            pa_b = team_bf_s1.predict_proba(Xt_s[:, bf_cols])[:, 1].astype(np.float64)
            db = (ph_b - pa_b) / np.maximum(ph_b + pa_b, 1e-6)
            db_q = np.round(np.clip(db, -0.9998, 0.9998) * 65536.0) / 65536.0
            p_ha_bf = np.clip(0.5 + 0.5 * db_q, 1e-4, 1.0 - 1e-4).astype(np.float32)
            # Continuous 2-12s (50..300f) missing-ball routing into Stage-1 p_ha1
            w_bf = np.clip((rb.gap_len.astype(np.float64) - 50.0) / 250.0, 0.0, 1.0) * (
                1.0 - 0.5 * rb.ball_conf.astype(np.float64)
            )
            d_blend = (1.0 - 0.65 * w_bf) * d_ha_q + (0.65 * w_bf) * db_q
            d_blend_q = np.round(np.clip(d_blend, -0.9998, 0.9998) * 65536.0) / 65536.0
            p_ha1 = np.clip(0.5 + 0.5 * d_blend_q, 1e-4, 1.0 - 1e-4).astype(np.float32)

        p_ha_nn, p_sw_nn = None, None
        neural_model = getattr(self, "neural_model", None)
        nn2 = getattr(self, "neural_v2", None)
        if nn2:
            from solution import neural_v2
            p_ha_nn, p_sw_nn = neural_v2.predict_match(nn2, m)
        elif neural_model is not None:
            from solution import neural_team
            p_ha_nn, p_sw_nn = neural_team.predict_neural_team(
                neural_model, m, rb=rb, vel=vel, Xtf=Xt_f, Xts=Xt_s
            )

        boundary_model = getattr(self, "boundary_model", None)
        switch_mode = getattr(self, "neural_switch_mode", "blend")
        if switch_mode not in ("blend", "ownership_only", "neural_only"):
            raise ValueError("Unknown neural switch mode: " + switch_mode)
        if boundary_model is not None:
            q = boundary_model.predict_from_features(Xt_f, Xt_s, m.blocks)
            if switch_mode == "ownership_only":
                p_sw_nn = q
            elif switch_mode != "neural_only":
                p_sw_nn = q if p_sw_nn is None else 0.5 * (q + p_sw_nn)
        elif switch_mode == "ownership_only":
            p_sw_nn = None

        return (
            rb,
            vel,
            p_d1,
            p_ha1,
            p_ha_bf,
            p_ha_nn,
            p_sw_nn,
            Xd[:, :40].astype(np.float32, copy=False),
            Xt_f[:, :65].astype(np.float32, copy=False),
            Xt_s[:, :65].astype(np.float32, copy=False),
        )

    def predict_stage1(self, m: Match) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, _, p_d1, p_ha1, _, _, _, Xd_base, Xf_base, Xs_base = self._compute_s1(m)
        return p_d1, p_ha1, Xd_base, Xf_base, Xs_base

    def predict_stage1_for_stacker(self, m: Match) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.predict_stage1(m)

    def _combine_stacker_and_neural(
        self,
        m: Match,
        p_d1: np.ndarray,
        p_ha1: np.ndarray,
        p_ha_nn: Optional[np.ndarray],
        p_sw_nn: Optional[np.ndarray],
        Xd_base: np.ndarray,
        Xf_base: np.ndarray,
        Xs_base: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.stacker is not None:
            lp = self.stacker.predict_log_proba(p_d1, p_ha1, Xd_base, Xf_base, m.blocks, Xs_base=Xs_base)
        else:
            live = 1.0 - p_d1.astype(np.float64)
            d_ha = 2.0 * p_ha1.astype(np.float64) - 1.0
            ph_live = np.clip(0.5 + 0.5 * d_ha, 1e-4, 1.0 - 1e-4)
            pa_live = np.clip(0.5 - 0.5 * d_ha, 1e-4, 1.0 - 1e-4)
            p3 = np.column_stack([live * ph_live, live * pa_live, p_d1])
            lp = np.log(np.clip(p3, 1e-5, 1.0)).astype(np.float32)

        w_nn = float(getattr(self, "neural_weight", 0.0))
        if p_ha_nn is not None and w_nn > 0.0:
            from solution.ownership_blend import blend_logp
            lp = blend_logp(lp, p_ha_nn, w_nn)

        sw_sig = p_sw_nn if p_sw_nn is not None else None
        return lp, sw_sig

    def predict_components(self, m: Match):
        rb, vel, pd1, pha1, pha_bf, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base = self._compute_s1(m)
        lp, sw_sig = self._combine_stacker_and_neural(m, pd1, pha1, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base)
        return rb, vel, pd1, pha1, pha_bf, pha_nn, psw_nn, lp, sw_sig

    def predict_log_proba(self, m: Match) -> np.ndarray:
        _, _, pd1, pha1, _, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base = self._compute_s1(m)
        lp, _ = self._combine_stacker_and_neural(m, pd1, pha1, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base)
        return lp

    def predict(self, m: Match) -> np.ndarray:
        _, _, pd1, pha1, _, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base = self._compute_s1(m)
        lp, sw_sig = self._combine_stacker_and_neural(m, pd1, pha1, pha_nn, psw_nn, Xd_base, Xf_base, Xs_base)
        if getattr(self, "freeze_dead_mask", False) or self.decode_cfg.get("freeze_dead_from_gbdt", False):
            base_lp, _ = self._combine_stacker_and_neural(m, pd1, pha1, None, None,
                                                         Xd_base, Xf_base, Xs_base)
            reference_cfg = dict(getattr(self, "dead_reference_cfg", None) or self.decode_cfg)
            reference_cfg.pop("freeze_dead_from_gbdt", None)
            reference = viterbi4.decode4(base_lp, m.blocks, reference_cfg)
            cfg = dict(viterbi4.DECODE4_DEFAULTS, **self.decode_cfg)
            return viterbi4.ownership_pass(float(cfg["w"]) * lp, reference, m.blocks,
                                           float(cfg["c_team"]) * 0.5,
                                           sw_signal=sw_sig, c_sw_gain=float(cfg.get("c_sw_gain", 0)) * 0.5)
        codes = viterbi4.decode4(lp, m.blocks, self.decode_cfg, sw_signal=sw_sig)
        snap_cfg = getattr(self, "line_snap_cfg", None)
        if snap_cfg is not None:
            from solution import line_snap
            codes = line_snap.snap_onsets(codes, m.ball, m.pitch, m.blocks, snap_cfg)
        return codes

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PipelineV4":
        with open(path, "rb") as f:
            return pickle.load(f)

