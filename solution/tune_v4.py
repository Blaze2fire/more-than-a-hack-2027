"""Decoder-only cross-validation on external OOF prediction caches.

Reads out-of-sample LOMO `.npz` caches produced by `solution.ablate_v4 --save-oof`,
evaluates candidate Viterbi4 parameter sets in parallel with structural invariant
`c_res_same <= c_res_flip`, and exports a development-selected decoder config.

This is NOT full nested pipeline validation: the model ancestors of each OOF
file must also exclude an outer test fixture to support that claim. Historical
ablate_v4 caches do not satisfy it. Use an independent test set after selection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from solution import viterbi4
from solution.common import D, H, ROOT, A, X

if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


DFL_GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
TUNE_SCENARIOS = ("natural", "atd_real", "atd_blind")

# Module-level worker globals for fast fork-free / clean ProcessPool execution
_WORKER_CASES: Dict[str, dict] = {}


def _load_cases(oof_dir: str, scenarios: Tuple[str, ...]) -> Dict[str, dict]:
    cases = {}
    for k, mid in enumerate(DFL_GAMES):
        for sc in scenarios:
            p = os.path.join(oof_dir, f"dfl_{mid}__{sc}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=False)
            cases[f"{k}__{sc}"] = {
                "fold": k,
                "game": f"dfl_{mid}",
                "scenario": sc,
                "lp": z["lp"].astype(np.float32),
                "sw_sig": z["sw_sig"].astype(np.float32) if z["sw_sig"].size > 0 else None,
                "gt": z["gt"].astype(np.int8),
                "gt_merged": z["gt_merged"].astype(np.int8),
                "blocks": [tuple(b) for b in z["blocks"].tolist()],
            }
    return cases


def _init_worker(cases_arg):
    global _WORKER_CASES
    _WORKER_CASES = cases_arg


def _eval_one_cfg_all_cases(args_tuple):
    idx, cfg = args_tuple
    scores = {}
    for key, c in _WORKER_CASES.items():
        pred = viterbi4.decode4(c["lp"], c["blocks"], cfg, sw_signal=c["sw_sig"])
        r_raw = pm.evaluate(c["gt"], pred, taus=(), bootstrap_n=0)
        r_mrg = pm.evaluate(c["gt_merged"], pred, taus=(), bootstrap_n=0)
        sf1 = float(r_raw["spell_f1_mean"])
        sf1_m = float(r_mrg["spell_f1_mean"])
        mf1 = float(r_raw["macro_f1"])
        n_gt = max(1, int(r_raw["n_gt_spells"]))
        pr = float(r_raw["n_pred_spells"]) / n_gt
        obj = 0.5 * (sf1 + sf1_m) + 0.05 * mf1
        scores[key] = (obj, sf1, sf1_m, mf1, pr)
    return idx, scores


def _sample_candidates(rng: np.random.Generator, n_random: int, seed_cfg: dict) -> List[dict]:
    cands = [dict(seed_cfg)]
    # Ensure structural sanity even in seed copy variant
    s_fix = dict(seed_cfg)
    s_fix["c_res_same"] = min(float(s_fix.get("c_res_same", 12.0)), float(s_fix.get("c_res_flip", 15.0)))
    s_fix.setdefault("c_sw_gain", 0.0)
    s_fix.setdefault("snap_win", 0)
    cands.append(s_fix)

    for _ in range(n_random):
        c_team = float(rng.uniform(2.0, 6.2))
        sw_frac = float(rng.choice([0.0, 0.20, 0.35, 0.45, 0.55, 0.65])) if rng.random() < 0.85 else float(rng.uniform(0.0, 0.65))
        c_res_flip = float(rng.uniform(7.5, 24.0))
        c_res_same = float(rng.uniform(0.45 * c_res_flip, c_res_flip))
        cands.append({
            "w": float(rng.uniform(0.55, 1.15)),
            "d_bias": float(rng.uniform(-0.22, 0.18)),
            "c_team": c_team,
            "c_sw_gain": float(sw_frac * c_team),
            "c_stop": float(rng.uniform(12.0, 36.0)),
            "c_res_same": c_res_same,
            "c_res_flip": c_res_flip,
            "min_dead": int(rng.integers(20, 64)),
            "min_live_blip": int(rng.integers(10, 30)),
            "min_turn": int(rng.integers(4, 15)),
            "snap_win": int(rng.choice([0, 3, 5, 8])),
        })
    return cands


def _local_neighbors(base: dict, rng: np.random.Generator, n_nbr: int = 24) -> List[dict]:
    nbrs = []
    for _ in range(n_nbr):
        c = dict(base)
        c["w"] = float(np.clip(c["w"] + rng.normal(0, 0.08), 0.45, 1.35))
        c["d_bias"] = float(np.clip(c["d_bias"] + rng.normal(0, 0.04), -0.30, 0.25))
        c["c_team"] = float(np.clip(c["c_team"] + rng.normal(0, 0.5), 1.5, 7.5))
        c["c_sw_gain"] = float(np.clip(c.get("c_sw_gain", 0.0) + rng.normal(0, 0.45), 0.0, 0.72 * c["c_team"]))
        c["c_stop"] = float(np.clip(c["c_stop"] + rng.normal(0, 2.5), 8.0, 42.0))
        c["c_res_flip"] = float(np.clip(c["c_res_flip"] + rng.normal(0, 2.0), 6.0, 30.0))
        c["c_res_same"] = float(np.clip(c["c_res_same"] + rng.normal(0, 1.5), 3.0, c["c_res_flip"]))
        c["min_dead"] = int(np.clip(c["min_dead"] + int(rng.integers(-6, 7)), 15, 70))
        c["min_live_blip"] = int(np.clip(c["min_live_blip"] + int(rng.integers(-4, 5)), 6, 35))
        c["min_turn"] = int(np.clip(c["min_turn"] + int(rng.integers(-3, 4)), 3, 18))
        c["snap_win"] = int(np.clip(c.get("snap_win", 0) + int(rng.integers(-2, 3)), 0, 12))
        nbrs.append(c)
    return nbrs


def _mean_obj(score_dict: dict, folds: List[int], scenarios: Tuple[str, ...]) -> Tuple[float, float, float, float]:
    vals = [score_dict[f"{k}__{sc}"] for k in folds for sc in scenarios if f"{k}__{sc}" in score_dict]
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()), float(arr[:, 3].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-dir", required=True)
    p.add_argument("--seed-cfg", default=os.path.join(ROOT, "solution/configs/viterbi4.json"))
    p.add_argument("--out-cfg", default=os.path.join(ROOT, "solution/configs/viterbi4_tuned.json"))
    p.add_argument("--n-random", type=int, default=80)
    p.add_argument("--n-local", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    cases = _load_cases(args.oof_dir, TUNE_SCENARIOS)
    assert cases, f"No DFL cases found in {args.oof_dir}"

    with open(args.seed_cfg) as f:
        seed_cfg = json.load(f)

    rng = np.random.default_rng(args.seed)
    cands = _sample_candidates(rng, args.n_random, seed_cfg)
    print(f"Evaluating {len(cands)} global Viterbi4 candidates across {len(cases)} DFL held-out cases ({args.workers} workers)...", flush=True)

    all_scores = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(cases,)) as ex:
        for _, sc_map in ex.map(_eval_one_cfg_all_cases, [(i, c) for i, c in enumerate(cands)]):
            all_scores.append(sc_map)

        all_folds = sorted({c["fold"] for c in cases.values()})
        ranked = sorted(
            [(i, _mean_obj(all_scores[i], all_folds, TUNE_SCENARIOS)[0]) for i in range(len(cands))],
            key=lambda x: -x[1],
        )
        best_global = cands[ranked[0][0]]

        # Candidate generation must not adapt to the folds later called held out.
        # Neighbors of an all-fold winner leaked those labels into candidate
        # construction. Refine around the fixed seed instead.
        nbrs = _local_neighbors(seed_cfg, rng, n_nbr=args.n_local)
        print(f"[{time.time() - t0:.1f}s] Evaluating {len(nbrs)} seed-centered neighbors...", flush=True)
        base_n = len(cands)
        cands.extend(nbrs)
        for _, sc_map in ex.map(_eval_one_cfg_all_cases, [(base_n + i, c) for i, c in enumerate(nbrs)]):
            all_scores.append(sc_map)

    # Decoder-only leave-one-fixture-out selection; see model-ancestor caveat above.
    nested_raw, nested_mrg, nested_mf1, nested_pr = [], [], [], []
    baseline_raw, baseline_mrg, baseline_mf1 = [], [], []
    per_fold_picks = []

    for hold_k in all_folds:
        tr_folds = [k for k in all_folds if k != hold_k]
        best_i = max(range(len(cands)), key=lambda i: _mean_obj(all_scores[i], tr_folds, TUNE_SCENARIOS)[0])
        per_fold_picks.append(cands[best_i])
        ho_obj, ho_raw, ho_mrg, ho_mf1 = _mean_obj(all_scores[best_i], [hold_k], TUNE_SCENARIOS)
        b_obj, b_raw, b_mrg, b_mf1 = _mean_obj(all_scores[0], [hold_k], TUNE_SCENARIOS)
        nested_raw.append(ho_raw)
        nested_mrg.append(ho_mrg)
        nested_mf1.append(ho_mf1)
        baseline_raw.append(b_raw)
        baseline_mrg.append(b_mrg)
        baseline_mf1.append(b_mf1)

    print(
        f"[{time.time() - t0:.1f}s] Decoder-only CV (not nested pipeline CV; nat+atd_real+atd_blind):\n"
        f"  Seed E014 cfg : raw={np.mean(baseline_raw):.4f}, merged={np.mean(baseline_mrg):.4f}, macro_f1={np.mean(baseline_mf1):.4f}\n"
        f"  Tuned decoder : raw={np.mean(nested_raw):.4f}, merged={np.mean(nested_mrg):.4f}, macro_f1={np.mean(nested_mf1):.4f}",
        flush=True,
    )

    best_overall_i = max(range(len(cands)), key=lambda i: _mean_obj(all_scores[i], all_folds, TUNE_SCENARIOS)[0])
    consensus_cfg = dict(cands[best_overall_i])
    os.makedirs(os.path.dirname(args.out_cfg), exist_ok=True)
    with open(args.out_cfg, "w") as f:
        json.dump(consensus_cfg, f, indent=2)
    with open(args.out_cfg + ".report.json", "w") as f:
        json.dump({"protocol": "decoder-only OOF development CV, not full nested pipeline CV",
                   "candidate_generation": "fixed seed and independent RNG; no all-fold adaptive refinement",
                   "source": os.path.abspath(args.oof_dir), "args": vars(args),
                   "fold_configs": per_fold_picks,
                   "baseline_raw": baseline_raw, "selected_raw": nested_raw,
                   "baseline_merged": baseline_mrg, "selected_merged": nested_mrg,
                   "config": consensus_cfg}, f, indent=2)
    print("Saved consensus tuned Viterbi4 config to", args.out_cfg, ":", json.dumps(consensus_cfg), flush=True)


if __name__ == "__main__":
    main()
