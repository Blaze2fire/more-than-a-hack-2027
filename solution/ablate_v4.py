"""Historical exploratory master-cache ablation harness (not strict nested LOMO).

Audit 2026-11-17: Stage-2 training rows inherit outer-test information through
their Stage-1 generators. Groningen is also present in the shared feature
imputer, and --no-metrica does not remove Metrica from that imputer. Existing
scores are development diagnostics only. Use deadline_benchmark for independent
fixture-group validation; this script requires explicit --allow-exploratory.

Usage:
    .venv/bin/python -m solution.ablate_v4 --tag E014_baseline --sc-limit 8 --save-oof scratch/oof_E014
"""
import argparse
import gc
from hashlib import sha256
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from solution import ball_tracker, bench_v4, datasets, external, features_v4, pipeline_v4, temporal, viterbi4
from solution.common import A, D, FPS, H, ROOT, X, Match

if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


DEFAULT_RESULT_DIR = os.path.join(ROOT, "solution", "out")


class _MatchStub:
    """Lightweight container exposing only gt/blocks/meta for Stage-2 temporal OOF stacking."""

    def __init__(self, gt: np.ndarray, blocks: List[Tuple[int, int]], meta: dict):
        self.gt = gt
        self.blocks = blocks
        self.meta = meta

    @property
    def n(self) -> int:
        return len(self.gt)


def elvern_burst_mask(
    length: int,
    blocks: List[Tuple[int, int]],
    fixture: str = "J03WPY",
    condition: str = "gap-stress-burst-01",
) -> np.ndarray:
    """Reproduce Elvern's exact SHA-256-seeded 12/18/24s burst stress mask."""
    seed = int.from_bytes(sha256(f"{fixture}/{condition}".encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    hidden = np.zeros(length, dtype=bool)
    for start, stop in blocks:
        cursor = start + int(rng.integers(12 * FPS, 24 * FPS + 1))
        while cursor < stop:
            width = int(rng.choice((12, 18, 24))) * FPS
            hidden[cursor : min(stop, cursor + width)] = True
            cursor += width + int(rng.integers(12 * FPS, 24 * FPS + 1))
    return hidden


def make_elvern_burst_match(m_nat: Match, fixture: str, condition: str = "gap-stress-burst-01") -> Match:
    hidden = elvern_burst_mask(m_nat.n, m_nat.blocks, fixture=fixture, condition=condition)
    ball_burst = m_nat.ball.copy()
    ball_burst[hidden] = np.nan
    return Match(
        name=m_nat.name,
        ball=ball_burst,
        pos=m_nat.pos.copy(),
        spd=m_nat.spd.copy(),
        team=m_nat.team.copy(),
        blocks=list(m_nat.blocks),
        pitch=m_nat.pitch,
        gt=m_nat.gt.copy() if m_nat.gt is not None else None,
        meta=dict(m_nat.meta, scenario=condition),
    )


def compute_short_and_trans_metrics(pred: np.ndarray, gt: np.ndarray, blocks: List[Tuple[int, int]]) -> dict:
    """Short-spell recall at IoU 0.5 (<=2s and <=1s) + transition F1/latency at tau=12,25."""
    valid = np.zeros(len(gt), dtype=bool)
    for a, b in blocks:
        valid[a:b] = True
    gt_eval = np.where(valid, gt, X).astype(np.int8)
    pred_eval = np.where(valid, pred, X).astype(np.int8)
    runs = pm.valid_runs(gt_eval)
    filled = pm.fill_abstentions(pred_eval, runs)

    refs = pm.possession_spells(gt_eval, runs)
    preds = pm.possession_spells(filled, runs)
    durs = np.array([e - s for s, e, _ in refs], dtype=np.int32)
    matched = pm.matched_gt_flags(pm.spell_overlaps(refs, preds), len(refs), 0.5)

    m2 = durs <= 50
    m1 = durs <= 25
    res_t = pm.evaluate(gt_eval, filled, taus=(12, 25), bootstrap_n=0)
    t12 = res_t["transitions"][12]
    t25 = res_t["transitions"][25]
    return {
        "short_2s_rec": float(matched[m2].mean()) if m2.any() else 0.0,
        "short_2s_matched": int(matched[m2].sum()),
        "short_2s_total": int(m2.sum()),
        "sub1s_rec": float(matched[m1].mean()) if m1.any() else 0.0,
        "sub1s_matched": int(matched[m1].sum()),
        "sub1s_total": int(m1.sum()),
        "trans_f1_12": float(t12["f1"]),
        "trans_f1_25": float(t25["f1"]),
        "med_lat_12": float(t12["median_latency"]) if t12["median_latency"] is not None else np.nan,
    }


def eval_match_full(
    pipe: pipeline_v4.PipelineV4,
    m: Match,
    scenario: str,
    decode_cfg: dict,
    oof_dir: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Predict, decode, and collect all official + regime + short-spell metrics on one Match."""
    rb, vel, pd1, pha1, pha_bf, pha_nn, psw_nn, lp, sw_sig = pipe.predict_components(m)
    pred = viterbi4.decode4(lp, m.blocks, decode_cfg, sw_signal=sw_sig)

    gt_merged = bench_v4.merge_uncontrolled_gt(m)
    row = bench_v4.score_match(m.name, scenario, pred, m.gt, gt_merged=gt_merged)
    row.update(compute_short_and_trans_metrics(pred, m.gt, m.blocks))

    diag = bench_v4.oracle_breakdown(
        m,
        lp,
        lambda lpx, bl: viterbi4.decode4(lpx, bl, decode_cfg, sw_signal=sw_sig),
    )
    row["oracle_dead"] = float(diag["oracle_dead_spell_f1"])
    row["oracle_team"] = float(diag["oracle_team_spell_f1"])
    for rk in ("vis", "short", "burst"):
        rinfo = diag["regimes"].get(rk, {})
        row[f"{rk}_team_acc"] = float(rinfo.get("team_acc", np.nan))
        row[f"{rk}_ad_acc"] = float(rinfo.get("ad_acc", np.nan))

    if oof_dir:
        os.makedirs(oof_dir, exist_ok=True)
        fname = f"{m.name}__{scenario}.npz"
        np.savez(
            os.path.join(oof_dir, fname),
            lp=lp.astype(np.float32),
            p_d1=pd1.astype(np.float32),
            p_ha1=pha1.astype(np.float32),
            p_ha_bf=pha_bf.astype(np.float32) if pha_bf is not None else np.array([], dtype=np.float32),
            p_ha_nn=pha_nn.astype(np.float32) if pha_nn is not None else np.array([], dtype=np.float32),
            p_sw_nn=psw_nn.astype(np.float32) if psw_nn is not None else np.array([], dtype=np.float32),
            sw_sig=sw_sig.astype(np.float32) if sw_sig is not None else np.array([], dtype=np.float32),
            rb_conf=rb.ball_conf.astype(np.float32),
            rb_gap=rb.gap_len.astype(np.int32),
            gt=m.gt.astype(np.int8),
            gt_merged=gt_merged.astype(np.int8),
            blocks=np.array(m.blocks, dtype=np.int32),
        )
    return row, diag


def check_invariants(pipe: pipeline_v4.PipelineV4, m_ref: Match) -> dict:
    """Verify dead_leak == 0.0 and swap_diff == 0.0 on a real held-out match."""
    m_nan = Match(
        name=m_ref.name + "_nanball",
        ball=np.full_like(m_ref.ball, np.nan),
        pos=m_ref.pos,
        spd=m_ref.spd,
        team=m_ref.team,
        blocks=m_ref.blocks,
        pitch=m_ref.pitch,
        gt=m_ref.gt,
        meta=m_ref.meta,
    )
    lp1 = pipe.predict_log_proba(m_ref)
    lp_nan = pipe.predict_log_proba(m_nan)
    dead_leak = float(np.max(np.abs(lp1[:, D] - lp_nan[:, D])))

    team_sw = np.where(m_ref.team == H, A, np.where(m_ref.team == A, H, m_ref.team)).astype(np.int8)
    m_sw = Match(
        name=m_ref.name + "_sw",
        ball=m_ref.ball,
        pos=m_ref.pos,
        spd=m_ref.spd,
        team=team_sw,
        blocks=m_ref.blocks,
        pitch=m_ref.pitch,
        gt=m_ref.gt,
        meta=m_ref.meta,
    )
    lp_sw = pipe.predict_log_proba(m_sw)
    swap_diff = float(np.max(np.abs(lp1[:, [A, H, D]] - lp_sw)))
    return {"dead_leak": dead_leak, "swap_diff": swap_diff}


def _f16_safe(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=6e4, neginf=-6e4), -6e4, 6e4).astype(np.float16)


def build_or_load_master_cache(
    raw_dfl: List[Match],
    gron_clean: Match,
    met_clean: List[Match],
    sc_mids: List[str],
    cache_dir: str,
) -> dict:
    """Extract and persist ONE unified master cache covering P1/P3/P4 variants cleanly."""
    meta_path = os.path.join(cache_dir, "master_meta.pkl")
    if os.path.exists(meta_path):
        print(f"[Master Cache HIT] Loading {cache_dir}...", flush=True)
        with open(meta_path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Master Cache MISS] Building unified master cache at {cache_dir}...", flush=True)

    imp_dfl = ball_tracker.fit_constellation_imputer([gron_clean] + list(met_clean), stride=6, seed=0)
    imp_gron = ball_tracker.fit_constellation_imputer(list(raw_dfl) + list(met_clean), stride=6, seed=0)

    specs = []
    # DFL 0..6
    for k, m in enumerate(raw_dfl):
        for si, sc in enumerate(("natural", "atd_real")):
            seed = 1000 + 10 * k + si
            specs.append((f"{m.name}/{sc}", k, "dfl", True, 4, lambda mr=m, s=sc, sd=seed: datasets.degrade_atd(mr, scenario=s, seed=sd)))
        # Dedicated 10-30s long-burst augmented training copy
        seed_lb = 1500 + 10 * k
        specs.append((
            f"{m.name}/long_burst", k, "dfl_lb", True, 4,
            lambda mr=m, sd=seed_lb: datasets.augment_long_burst_gaps(
                datasets.degrade_atd(mr, scenario="atd_real", seed=sd), seed=sd + 50, prob=0.85
            ),
        ))

    # Groningen
    for si, sc in enumerate(("natural", "atd_real")):
        seed = 2000 + si
        specs.append((f"gron/{sc}", si % 2, "gron", True, 4, lambda mr=gron_clean, s=sc, sd=seed: datasets.degrade_atd(mr, scenario=s, seed=sd)))
    specs.append((
        "gron/long_burst", 2, "gron_lb", True, 4,
        lambda mr=gron_clean: datasets.augment_long_burst_gaps(
            datasets.degrade_atd(mr, scenario="atd_real", seed=2500), seed=2550, prob=0.85
        ),
    ))

    # SkillCorner Current
    for i, mid in enumerate(sc_mids):
        f_sc = i % 7
        specs.append((f"sc_{mid}/natural", f_sc, "sc", False, 6, lambda md=mid: datasets.load_skillcorner_current(md)))
        specs.append((f"sc_{mid}/atd_real", f_sc, "sc", False, 6, lambda md=mid, sd=3000 + i: datasets.degrade_atd(datasets.load_skillcorner_current(md), scenario="atd_real", seed=sd)))
        if i % 2 == 0:
            specs.append((
                f"sc_{mid}/long_burst", f_sc, "sc_lb", False, 6,
                lambda md=mid, sd=3500 + i: datasets.augment_long_burst_gaps(
                    datasets.degrade_atd(datasets.load_skillcorner_current(md), scenario="atd_real", seed=sd), seed=sd + 50, prob=0.85
                ),
            ))

    # Metrica (both external.degrade and datasets.degrade_atd variants for P4 ablation)
    sc_map = {"path": "natural", "hold": "atd_real", "hidden": "atd_carried_dead"}
    for k, m in enumerate(met_clean):
        f_met = k % 7
        for si, s in enumerate(("path", "hold", "hidden")):
            seed = 4000 + 10 * k + si
            specs.append((f"{m.name}/{s}_ext", f_met, "met_ext", True, 4, lambda mr=m, sc=s: external.degrade(mr, sc, seed=0)))
            specs.append((f"{m.name}/{s}_atd", f_met, "met_atd", True, 4, lambda mr=m, sc=sc_map[s], sd=seed: datasets.degrade_atd(mr, scenario=sc, seed=sd)))

    from solution import neural_team
    records = []
    nn_windows_by_rec = {}
    rng_nn = np.random.default_rng(77)

    for idx, (label, fold, src, include_dead, stride, builder) in enumerate(specs):
        m = builder()
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
        rb = ball_tracker.reconstruct_ball(m, imputer=imp_dfl, vel=vel)
        gt = m.gt
        mask = np.zeros(m.n, dtype=bool)
        for b0, b1 in m.blocks:
            mask[b0:b1:stride] = True
        bw = pipeline_v4.boundary_weights(m)

        Xd = features_v4.build_dead_features(m, vel=vel) if (include_dead and m.meta.get("has_real_dead_ball", True)) else None
        Xt_f = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=True)
        Xt_s = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=True)

        dm = (mask & (gt != X)) if Xd is not None else np.zeros(m.n, dtype=bool)
        lm = mask & ((gt == H) | (gt == A))

        Xd_row = Xd[dm].astype(np.float32) if dm.any() else None
        yd_row = (gt[dm] == D).astype(np.int8) if dm.any() else None
        wd_row = bw[dm].astype(np.float32) if dm.any() else None

        if lm.any():
            Xt_row = np.vstack([Xt_f[lm], Xt_s[lm]]).astype(np.float32)
            yt_row = np.concatenate([(gt[lm] == H).astype(np.int8), (gt[lm] == A).astype(np.int8)])
            wt_row = np.concatenate([bw[lm], bw[lm]]).astype(np.float32)
        else:
            Xt_row = yt_row = wt_row = None

        # Always extract neural windows so E015..E018 reuse cache in 0s
        nn_windows_by_rec[idx] = neural_team.extract_match_neural_windows(
            m, rb, vel, Xt_f, Xt_s, include_dead=include_dead, win_len=1500, max_wins=6, rng=rng_nn
        )

        # Save compact float16 Stage-2 matrices for non-long-burst matches (plus SC for P4b)
        s2_path = None
        if not src.endswith("_lb"):
            s2_path = os.path.join(cache_dir, f"s2_{idx:03d}.npz")
            np.savez(
                s2_path,
                Xd=_f16_safe(Xd) if Xd is not None else np.empty((0, 0), dtype=np.float16),
                Xtf=_f16_safe(Xt_f),
                Xts=_f16_safe(Xt_s),
                gt=gt.astype(np.int8),
                blocks=np.array(m.blocks, dtype=np.int32),
                rb_conf=rb.ball_conf.astype(np.float32),
                rb_gap=rb.gap_len.astype(np.int32),
                meta_json=np.array([json.dumps(m.meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))]),
            )

        records.append({
            "idx": idx,
            "label": label,
            "fold": fold,
            "src": src,
            "include_dead": include_dead,
            "Xd_row": Xd_row,
            "yd_row": yd_row,
            "wd_row": wd_row,
            "Xt_row": Xt_row,
            "yt_row": yt_row,
            "wt_row": wt_row,
            "s2_path": s2_path,
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(specs):
            print(f"  [{time.time() - t0:.1f}s] Cached {idx + 1}/{len(specs)} specs ({label})", flush=True)
        del Xd, Xt_f, Xt_s, rb, vel, m

    nn_path = os.path.join(cache_dir, "nn_windows.pkl")
    with open(nn_path, "wb") as f:
        pickle.dump(nn_windows_by_rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    del nn_windows_by_rec

    bundle = {"imp_dfl": imp_dfl, "imp_gron": imp_gron, "records": records, "nn_path": nn_path}
    with open(meta_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
    print(f"[Master Cache READY] {len(records)} specs built in {time.time() - t0:.1f}s.", flush=True)
    return bundle


def _active_records(records: List[dict], args) -> List[dict]:
    active = []
    for r in records:
        src = r["src"]
        if src in ("dfl", "gron", "sc"):
            active.append(r)
        elif src in ("dfl_lb", "gron_lb", "sc_lb") and args.long_burst:
            active.append(r)
        elif src == "met_ext" and (not args.no_metrica) and args.metrica_degrade == "external":
            active.append(r)
        elif src == "met_atd" and (not args.no_metrica) and args.metrica_degrade == "atd":
            active.append(r)
    return active


def _stack_rows(recs: List[dict], key_x: str, key_y: str, key_w: str, x_cols: Optional[int] = None):
    xs = [r[key_x][:, :x_cols] if (x_cols is not None and r[key_x] is not None) else r[key_x] for r in recs if r[key_x] is not None]
    ys = [r[key_y] for r in recs if r[key_y] is not None]
    ws = [r[key_w] for r in recs if r[key_w] is not None]
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def _compute_s2_rows_for_record(d1_f, t1_f, tbf_f, bf_cols, rec: dict, n_t_cols: int, sc_s2_weight: float) -> dict:
    """Generate OOF Stage-1 predictions & extract Stage-2 sequence training arrays ONCE per record."""
    z = np.load(rec["s2_path"], allow_pickle=False)
    gt = z["gt"]
    blocks = [tuple(b) for b in z["blocks"].tolist()]
    meta = json.loads(str(z["meta_json"][0]))

    Xtf = z["Xtf"][:, :n_t_cols].astype(np.float32)
    Xts = z["Xts"][:, :n_t_cols].astype(np.float32)
    Xd_raw = z["Xd"]
    if len(Xd_raw) > 0:
        Xd = Xd_raw.astype(np.float32)
        pd1 = np.clip(d1_f.predict_proba(Xd)[:, 1], 1e-4, 1.0 - 1e-4).astype(np.float32)
        Xd_base = Xd[:, :40].astype(np.float32)
    else:
        pd1 = np.full(len(Xtf), 0.1, dtype=np.float32)
        Xd_base = np.zeros((len(Xtf), 40), dtype=np.float32)

    ph = t1_f.predict_proba(Xtf)[:, 1].astype(np.float64)
    pa = t1_f.predict_proba(Xts)[:, 1].astype(np.float64)
    d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
    d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
    if tbf_f is not None and bf_cols is not None:
        ph_b = tbf_f.predict_proba(Xtf[:, bf_cols])[:, 1].astype(np.float64)
        pa_b = tbf_f.predict_proba(Xts[:, bf_cols])[:, 1].astype(np.float64)
        db = (ph_b - pa_b) / np.maximum(ph_b + pa_b, 1e-6)
        db_q = np.round(np.clip(db, -0.9998, 0.9998) * 65536.0) / 65536.0
        gap = z["rb_gap"].astype(np.float64)
        conf = z["rb_conf"].astype(np.float64)
        w_bf = np.clip((gap - 50.0) / 250.0, 0.0, 1.0) * (1.0 - 0.5 * conf)
        d_ha_q = np.round(np.clip((1.0 - 0.65 * w_bf) * d_ha_q + 0.65 * w_bf * db_q, -0.9998, 0.9998) * 65536.0) / 65536.0
    pha1 = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)

    stub = _MatchStub(gt=gt, blocks=blocks, meta=meta)
    mask = np.zeros(len(gt), dtype=bool)
    for b0, b1 in blocks:
        mask[b0:b1:3] = True
    bw = pipeline_v4.boundary_weights(stub)

    Zd_row = yd_row = wd_row = None
    if rec["include_dead"] and meta.get("has_real_dead_ball", True):
        dm = mask & (gt != X)
        if dm.any():
            Zd = temporal.build_dead_seq_features(pd1, Xd_base, blocks)
            Zd_row = Zd[dm].astype(np.float32)
            yd_row = (gt[dm] == D).astype(np.int8)
            wd_row = bw[dm].astype(np.float32)

    Zt_row = yt_row = wt_row = None
    lm = mask & ((gt == H) | (gt == A))
    if lm.any():
        s2_w = 1.0 if rec["include_dead"] else float(sc_s2_weight)
        Xf_base = Xtf[:, :65]
        Xs_base = Xts[:, :65]
        Zt_f = temporal.build_team_seq_features(pha1, pd1, Xf_base, blocks, swap=False)
        Zt_s = temporal.build_team_seq_features(pha1, pd1, Xf_base, blocks, swap=True, Xs_base=Xs_base)
        wt_lm = bw[lm] * s2_w
        Zt_row = np.vstack([Zt_f[lm], Zt_s[lm]]).astype(np.float32)
        yt_row = np.concatenate([(gt[lm] == H).astype(np.int8), (gt[lm] == A).astype(np.int8)])
        wt_row = np.concatenate([wt_lm, wt_lm]).astype(np.float32)

    return {
        "fold": rec["fold"],
        "src": rec["src"],
        "Zd_row": Zd_row,
        "yd_row": yd_row,
        "wd_row": wd_row,
        "Zt_row": Zt_row,
        "yt_row": yt_row,
        "wt_row": wt_row,
    }


def aggregate_summary(df: pd.DataFrame) -> dict:
    summary = {}
    dfl_df = df[df["game"].str.startswith("dfl_") & ~df["scenario"].isin(["gap-stress-burst-01"])]
    for sc in ("natural", "atd_real", "atd_blind", "atd_carried_dead"):
        sub = dfl_df[dfl_df["scenario"] == sc]
        if len(sub) == 0:
            continue
        summary[f"dfl_{sc}"] = {
            "spell_f1": float(sub["spell_f1"].mean()),
            "spell_f1_merged": float(sub["spell_f1_merged"].mean()),
            "macro_f1": float(sub["macro_f1"].mean()),
            "alive_dead_f1": float(sub["alive_dead_f1"].mean()),
            "team_acc": float(sub["team_acc"].mean()),
            "poss_ratio": float(sub["poss_ratio"].mean()),
            "short_2s_rec": float(sub["short_2s_rec"].mean()),
            "sub1s_rec": float(sub["sub1s_rec"].mean()),
            "trans_f1_12": float(sub["trans_f1_12"].mean()),
            "trans_f1_25": float(sub["trans_f1_25"].mean()),
            "vis_team_acc": float(sub["vis_team_acc"].mean()),
            "burst_team_acc": float(sub["burst_team_acc"].mean()),
            "oracle_dead": float(sub["oracle_dead"].mean()),
            "oracle_team": float(sub["oracle_team"].mean()),
        }
    for row in df[df["scenario"] == "gap-stress-burst-01"].to_dict("records"):
        summary[f"{row['game']}_burst01"] = {
            "spell_f1": float(row["spell_f1"]),
            "spell_f1_merged": float(row["spell_f1_merged"]),
            "macro_f1": float(row["macro_f1"]),
            "alive_dead_f1": float(row["alive_dead_f1"]),
            "team_acc": float(row["team_acc"]),
            "poss_ratio": float(row["poss_ratio"]),
            "short_2s_rec": float(row["short_2s_rec"]),
            "sub1s_rec": float(row["sub1s_rec"]),
            "burst_team_acc": float(row["burst_team_acc"]),
        }
    for row in df[df["game"] == "groningen"].to_dict("records"):
        summary[f"gron_{row['scenario']}"] = {
            "spell_f1": float(row["spell_f1"]),
            "spell_f1_merged": float(row["spell_f1_merged"]),
            "macro_f1": float(row["macro_f1"]),
            "alive_dead_f1": float(row["alive_dead_f1"]),
            "team_acc": float(row["team_acc"]),
            "poss_ratio": float(row["poss_ratio"]),
            "short_2s_rec": float(row["short_2s_rec"]),
            "burst_team_acc": float(row["burst_team_acc"]),
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="ens3", choices=("ens3", "lgbm", "xgb"))
    p.add_argument("--folds", default="all", help="Comma-separated DFL fold indices 0..6, or 'all'")
    p.add_argument("--no-gron", action="store_true")
    p.add_argument("--sc-limit", type=int, default=8)
    p.add_argument("--no-metrica", action="store_true")
    p.add_argument("--metrica-degrade", default="external", choices=("external", "atd"))
    p.add_argument("--sc-s2-weight", type=float, default=0.0)
    p.add_argument("--long-burst", action="store_true")
    p.add_argument("--use-neural", action="store_true")
    p.add_argument("--neural-epochs", type=int, default=14)
    p.add_argument("--neural-weight", type=float, default=0.30)
    p.add_argument("--use-p3", action="store_true")
    p.add_argument("--use-bf-specialist", action="store_true")
    p.add_argument("--decode-cfg", default=os.path.join(ROOT, "solution/configs/viterbi4.json"))
    p.add_argument("--master-cache", default="")
    p.add_argument("--save-oof", default="")
    p.add_argument("--out-json", default="")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--allow-exploratory", action="store_true",
                   help="Acknowledge historical model-ancestor leakage; results are not held-out estimates")
    args = p.parse_args()
    if not args.allow_exploratory:
        p.error("Historical cache has model-ancestor leakage. Use solution.deadline_benchmark, or explicitly acknowledge --allow-exploratory for diagnostics.")

    t0 = time.time()
    use_gpu = not args.no_gpu
    print(f"=== 7-Fold Cross-Fitted LOMO Ablation: {args.tag} ===", flush=True)
    with open(args.decode_cfg) as f:
        decode_cfg = json.load(f)

    cache_dir = args.master_cache or os.path.join(DEFAULT_RESULT_DIR, f"master_feat_cache_sc{args.sc_limit}")
    meta_path = os.path.join(cache_dir, "master_meta.pkl")
    if os.path.exists(meta_path):
        print(f"[Master Cache HIT] Loading {cache_dir}...", flush=True)
        with open(meta_path, "rb") as f:
            bundle = pickle.load(f)
    else:
        raw_dfl = datasets.load_all_dfl()
        gron_clean = datasets.load_groningen()
        met_clean = [datasets.standardize_match(external.load(g)) for g in external.GAMES]
        sc_mids = datasets.list_skillcorner_current()[: args.sc_limit]
        bundle = build_or_load_master_cache(raw_dfl, gron_clean, met_clean, sc_mids, cache_dir=cache_dir)
        del raw_dfl, gron_clean, met_clean
        gc.collect()

    imp_dfl = bundle["imp_dfl"]
    imp_gron = bundle["imp_gron"]
    nn_path = bundle["nn_path"]
    records = _active_records(bundle["records"], args)
    del bundle
    gc.collect()

    nn_wins_map = {}
    if args.use_neural:
        from solution import neural_team
        with open(nn_path, "rb") as f:
            nn_wins_map = pickle.load(f)

    n_t_cols = 211 if args.use_p3 else 156
    bf_cols = features_v4.ball_free_team_cols(n_t_cols) if args.use_bf_specialist else None

    # Step A & B: 7-Fold Cross-Fitted Stage-1 (+ Neural) & single-pass Stage-2 OOF rows
    print(f"\n[{time.time() - t0:.1f}s] --- Step A/B: 7-Fold Cross-Fitted Stage-1 & Stage-2 OOF ({len(records)} active records) ---", flush=True)
    fold_models = {}
    s2_rows = []

    for f_hold in range(7):
        tf = time.time()
        tr_recs = [r for r in records if r["fold"] != f_hold]
        oof_recs = [r for r in records if r["fold"] == f_hold and r["s2_path"] is not None and (r["include_dead"] or args.sc_s2_weight > 0.0)]

        Xd_tr, yd_tr, wd_tr = _stack_rows(tr_recs, "Xd_row", "yd_row", "wd_row")
        d1_f = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=10 + f_hold, use_gpu=use_gpu).fit(
            Xd_tr, yd_tr, sample_weight=wd_tr
        )
        del Xd_tr, yd_tr, wd_tr

        Xt_tr, yt_tr, wt_tr = _stack_rows(tr_recs, "Xt_row", "yt_row", "wt_row", x_cols=n_t_cols)
        t1_f = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=20 + f_hold, use_gpu=use_gpu).fit(
            Xt_tr, yt_tr, sample_weight=wt_tr
        )
        tbf_f = None
        if args.use_bf_specialist and bf_cols is not None:
            tbf_f = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=30 + f_hold, use_gpu=use_gpu).fit(
                Xt_tr[:, bf_cols], yt_tr, sample_weight=wt_tr
            )
        del Xt_tr, yt_tr, wt_tr

        nn_f = None
        if args.use_neural:
            nn_w_list = [w for r in tr_recs for w in nn_wins_map.get(r["idx"], [])]
            nn_f = neural_team.fit_neural_team_from_windows(
                nn_w_list, epochs=args.neural_epochs, seed=50 + f_hold, use_p3=args.use_p3, use_gpu=use_gpu
            )

        for r in oof_recs:
            s2_rows.append(_compute_s2_rows_for_record(d1_f, t1_f, tbf_f, bf_cols, r, n_t_cols, args.sc_s2_weight))

        fold_models[f_hold] = (d1_f, t1_f, tbf_f, nn_f)
        print(f"  [{time.time() - t0:.1f}s] Cross-fit fold {f_hold}/6 done in {time.time() - tf:.1f}s ({len(oof_recs)} OOF matches).", flush=True)

    # Also fit Groningen Zero-Shot Stage-1 while records are still loaded, then free all Stage-1 arrays!
    gron_s1 = None
    if not args.no_gron:
        tr_gron = [r for r in records if r["src"] not in ("gron", "gron_lb")]
        Xd_g, yd_g, wd_g = _stack_rows(tr_gron, "Xd_row", "yd_row", "wd_row")
        d1_g = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=900, use_gpu=use_gpu).fit(Xd_g, yd_g, sample_weight=wd_g)
        del Xd_g, yd_g, wd_g
        Xt_g, yt_g, wt_g = _stack_rows(tr_gron, "Xt_row", "yt_row", "wt_row", x_cols=n_t_cols)
        t1_g = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=901, use_gpu=use_gpu).fit(Xt_g, yt_g, sample_weight=wt_g)
        tbf_g = None
        if args.use_bf_specialist and bf_cols is not None:
            tbf_g = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=902, use_gpu=use_gpu).fit(
                Xt_g[:, bf_cols], yt_g, sample_weight=wt_g
            )
        del Xt_g, yt_g, wt_g
        nn_g = None
        if args.use_neural:
            nn_w_g = [w for r in tr_gron for w in nn_wins_map.get(r["idx"], [])]
            nn_g = neural_team.fit_neural_team_from_windows(
                nn_w_g, epochs=args.neural_epochs, seed=950, use_p3=args.use_p3, use_gpu=use_gpu
            )
        gron_s1 = (d1_g, t1_g, tbf_g, nn_g)

    del records, nn_wins_map
    gc.collect()

    # Step C: Fit Stage-2 Stacker per held-out DFL fold & evaluate held-out DFL match
    folds = list(range(len(datasets.DFL_GAMES))) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    rows = []
    invariants = {}
    oof_dir = args.save_oof if args.save_oof else None

    for k in folds:
        tf = time.time()
        m_raw = datasets.load_dfl(datasets.DFL_GAMES[k])
        d1_k, t1_k, tbf_k, nn_k = fold_models.pop(k)

        s2_tr = [s for s in s2_rows if s["fold"] != k]
        Zd_s2, yd_s2, wd_s2 = _stack_rows(s2_tr, "Zd_row", "yd_row", "wd_row")
        dead_s2 = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=100 + k, use_gpu=use_gpu).fit(
            Zd_s2, yd_s2, sample_weight=wd_s2
        )
        del Zd_s2, yd_s2, wd_s2

        Zt_s2, yt_s2, wt_s2 = _stack_rows(s2_tr, "Zt_row", "yt_row", "wt_row")
        team_s2 = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=200 + k, use_gpu=use_gpu).fit(
            Zt_s2, yt_s2, sample_weight=wt_s2
        )
        del Zt_s2, yt_s2, wt_s2

        stk_k = temporal.TemporalStacker(dead_s2=dead_s2, team_s2=team_s2)
        pipe = pipeline_v4.PipelineV4(
            imputer=imp_dfl,
            dead_s1=d1_k,
            team_s1=t1_k,
            stacker=stk_k,
            decode_cfg=decode_cfg,
            team_bf_s1=tbf_k,
            bf_cols=bf_cols,
            neural_model=nn_k,
            neural_weight=args.neural_weight,
            use_p3=args.use_p3,
        )

        for sc in ("natural", "atd_real", "atd_blind", "atd_carried_dead"):
            m_eval = datasets.degrade_atd(m_raw, scenario=sc, seed=100 + k)
            row, _ = eval_match_full(pipe, m_eval, sc, decode_cfg, oof_dir=oof_dir)
            rows.append(row)

        if m_raw.name in ("dfl_J03WPY", "dfl_J03WQQ", "dfl_J03WR9"):
            fixture = m_raw.name.replace("dfl_", "")
            m_nat = datasets.degrade_atd(m_raw, scenario="natural", seed=100 + k)
            m_burst = make_elvern_burst_match(m_nat, fixture=fixture, condition="gap-stress-burst-01")
            row_b, _ = eval_match_full(pipe, m_burst, "gap-stress-burst-01", decode_cfg, oof_dir=oof_dir)
            rows.append(row_b)

        if k == 4 or k == folds[0]:
            m_blind = datasets.degrade_atd(m_raw, scenario="atd_blind", seed=100 + k)
            inv = check_invariants(pipe, m_blind)
            invariants[m_raw.name] = inv
            print(f"    Invariants on {m_raw.name}: dead_leak={inv['dead_leak']:.6g}, swap_diff={inv['swap_diff']:.6g}", flush=True)

        print(f"  [{time.time() - t0:.1f}s] Evaluated held-out {m_raw.name} in {time.time() - tf:.1f}s.", flush=True)
        del pipe, stk_k, m_raw, d1_k, t1_k, tbf_k, nn_k
        gc.collect()

    fold_models.clear()

    # Step D: Groningen Zero-Shot (trained strictly on non-Groningen records)
    if not args.no_gron and gron_s1 is not None:
        tf = time.time()
        print(f"\n[{time.time() - t0:.1f}s] --- Step D: Groningen Zero-Shot ---", flush=True)
        d1_g, t1_g, tbf_g, nn_g = gron_s1

        s2_gron = [s for s in s2_rows if s["src"] != "gron"]
        Zd_s2, yd_s2, wd_s2 = _stack_rows(s2_gron, "Zd_row", "yd_row", "wd_row")
        ds2_g = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=910, use_gpu=use_gpu).fit(Zd_s2, yd_s2, sample_weight=wd_s2)
        del Zd_s2, yd_s2, wd_s2
        Zt_s2, yt_s2, wt_s2 = _stack_rows(s2_gron, "Zt_row", "yt_row", "wt_row")
        ts2_g = pipeline_v4.EnsembleBinaryClf(mode=args.mode, seed=911, use_gpu=use_gpu).fit(Zt_s2, yt_s2, sample_weight=wt_s2)
        del Zt_s2, yt_s2, wt_s2

        pipe_gron = pipeline_v4.PipelineV4(
            imputer=imp_gron,
            dead_s1=d1_g,
            team_s1=t1_g,
            stacker=temporal.TemporalStacker(dead_s2=ds2_g, team_s2=ts2_g),
            decode_cfg=decode_cfg,
            team_bf_s1=tbf_g,
            bf_cols=bf_cols,
            neural_model=nn_g,
            neural_weight=args.neural_weight,
            use_p3=args.use_p3,
        )
        gron_clean = datasets.load_groningen()
        for sc in ("natural", "atd_real"):
            gm = datasets.degrade_atd(gron_clean, scenario=sc, seed=199)
            row, _ = eval_match_full(pipe_gron, gm, sc, decode_cfg, oof_dir=oof_dir)
            rows.append(row)
        print(f"  [{time.time() - t0:.1f}s] Groningen zero-shot done in {time.time() - tf:.1f}s.", flush=True)

    df = pd.DataFrame(rows)
    summary = aggregate_summary(df)
    print(f"\n=== SUMMARY ({args.tag}, elapsed {time.time() - t0:.1f}s) ===", flush=True)
    for k_sum, v_sum in summary.items():
        print(f"  {k_sum:24s}: " + ", ".join(f"{mk}={mv:.4f}" if isinstance(mv, float) else f"{mk}={mv}" for mk, mv in v_sum.items()))

    out_path = args.out_json or os.path.join(DEFAULT_RESULT_DIR, f"ablation_{args.tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "args": vars(args),
                "invariants": invariants,
                "summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print("Saved ablation report to", out_path, flush=True)


if __name__ == "__main__":
    main()
