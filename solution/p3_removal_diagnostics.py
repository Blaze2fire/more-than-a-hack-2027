"""Frozen-model P3 module-removal diagnostics, not retrained causal ablations.

One cached feature extraction per external evaluation view. Feature-group removal
uses fixed medians estimated exclusively from DFL4 training fixtures. Removal can
create distribution shift, so differences do not establish standalone benefits.
"""
import argparse
import dataclasses
from hashlib import sha256
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from solution import ablate_v4, ball_tracker, bench_v4, datasets, features_v4, train_v4, viterbi4
from solution.ablate_sc_stage2 import load_baseline, write_json
from solution.common import A, D, H


def training_medians(pipe):
    """Reference feature values from labelled training fixtures, never test data."""
    sampled = []
    for mid in datasets.DFL_GAMES[:4]:
        raw = datasets.load_dfl(mid)
        for scenario in ("natural", "atd_real"):
            seed = train_v4.fixture_seed(raw.name, scenario, int(pipe.training_provenance["seed"]))
            m = datasets.degrade_atd(raw, scenario, seed)
            vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
            rb = ball_tracker.reconstruct_ball(m, imputer=pipe.imputer, vel=vel)
            for swap in (False, True):
                xf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=swap, use_p3=True)
                sampled.append(xf[::20, 156:211].copy())
        print(f"Training-only removal reference {mid}", flush=True)
    medians = np.median(np.concatenate(sampled), axis=0).astype(np.float32)
    if medians.shape != (55,) or not np.isfinite(medians).all():
        raise ValueError("Invalid training-only P3 reference")
    return medians


def run(args):
    start = time.monotonic()
    pipe, manifest = load_baseline(args.pipeline)
    if not getattr(pipe, "use_p3", False) or pipe.team_bf_s1 is None:
        raise ValueError("Requires a P3 model trained with the optional ball-free router")
    if getattr(pipe, "neural_model", None) is not None or getattr(pipe, "neural_v2", None):
        raise ValueError("Isolate P3 before any neural blend")
    if getattr(pipe, "boundary_model", None) is not None:
        raise ValueError("Isolate P3 before adding a boundary model")
    if getattr(pipe, "freeze_dead_mask", False) or pipe.decode_cfg.get("freeze_dead_from_gbdt"):
        raise ValueError("This diagnostic uses the model's original single-pass decoder")
    median = training_medians(pipe)
    variants = ("full", "no_fallback", "axes_neutral", "deblur_neutral")
    predeclared = dict(variants=variants, training_reference="DFL4 natural/atd_real, both orientations, stride20 median",
                       axes_columns=[156, 201], deblur_columns=[201, 211],
                       axes_medians=median[:45].tolist(), deblur_medians=median[45:].tolist(),
                       interpretation="frozen-model removal diagnostic with possible distribution shift; no causal retrain claim")
    write_json(str(args.out) + ".protocol.json", predeclared)
    rows, dead_diff, parity_cases, uncached_parity_cases = [], 0.0, 0, 0
    cases = [datasets.load_dfl(mid) for mid in datasets.DFL_GAMES[4:]] + [datasets.load_groningen()]
    for raw in cases:
        for scenario in ("natural", "atd_real", "gap-stress-burst-01"):
            seeds = (911, 2027) if scenario == "atd_real" else (911,)
            for seed in seeds:
                if scenario.startswith("gap-stress"):
                    natural = datasets.degrade_atd(raw, "natural", seed)
                    m = ablate_v4.make_elvern_burst_match(natural, raw.name.replace("dfl_", ""))
                else:
                    m = datasets.degrade_atd(raw, scenario, seed)
                vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
                rb = ball_tracker.reconstruct_ball(m, imputer=pipe.imputer, vel=vel)
                xd = features_v4.build_dead_features(m, vel=vel)
                xf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=True)
                xs = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=True)
                merged = bench_v4.merge_uncontrolled_gt(m)
                gap = bench_v4.gap_lengths(np.isfinite(m.ball[:, 0]), m.blocks)
                uncached_reference = pipe.predict_log_proba(m) if not rows else None
                for variant in variants:
                    vf, vs = xf, xs
                    candidate = pipe
                    if variant == "no_fallback":
                        candidate = dataclasses.replace(pipe, team_bf_s1=None, bf_cols=None)
                    if variant in ("axes_neutral", "deblur_neutral"):
                        vf, vs = xf.copy(), xs.copy()
                        cols = slice(156, 201) if variant == "axes_neutral" else slice(201, 211)
                        vals = median[:45] if variant == "axes_neutral" else median[45:]
                        vf[:, cols], vs[:, cols] = vals, vals
                    def cached_team_features(match, *a, **kw):
                        if match is not m:
                            raise ValueError("Cached feature inputs must be the same external view")
                        return vs if kw.get("swap", False) else vf
                    with patch.object(ball_tracker, "compute_player_vel", lambda *a, **k: vel), \
                         patch.object(ball_tracker, "reconstruct_ball", lambda *a, **k: rb), \
                         patch.object(features_v4, "build_dead_features", lambda *a, **k: xd), \
                         patch.object(features_v4, "build_team_features", cached_team_features):
                        lp = candidate.predict_log_proba(m)
                        pred = viterbi4.decode4(lp, m.blocks, candidate.decode_cfg)
                        if variant == "full":
                            np.testing.assert_array_equal(pred, candidate.predict(m))
                            parity_cases += 1
                            if uncached_reference is not None:
                                np.testing.assert_array_equal(lp, uncached_reference)
                                uncached_parity_cases += 1
                    if variant == "full":
                        reference = lp[:, D].copy()
                    else:
                        dead_diff = max(dead_diff, float(np.max(np.abs(reference - lp[:, D]))))
                        np.testing.assert_array_equal(reference, lp[:, D])
                    row = bench_v4.score_match(m.name, scenario, pred, m.gt, merged)
                    row.update(ablate_v4.compute_short_and_trans_metrics(pred, m.gt, m.blocks))
                    live = (m.gt == H) | (m.gt == A)
                    conditional = np.argmax(lp[:, :2], axis=1)
                    row["strict_team_acc"] = float(np.mean(pred[live] == m.gt[live]))
                    row["conditional_team_acc"] = float(np.mean(conditional[live] == m.gt[live]))
                    log_cond = lp[:, :2].astype(np.float64)
                    log_cond -= np.logaddexp(log_cond[:, 0], log_cond[:, 1])[:, None]
                    row["conditional_log_loss"] = float(np.mean(-log_cond[np.flatnonzero(live), m.gt[live]]))
                    live_burst = live & (gap > 25)
                    row["strict_burst_team_acc"] = float(np.mean(pred[live_burst] == m.gt[live_burst])) if live_burst.any() else None
                    row.update(variant=variant, seed=seed,
                               provider="DFL" if raw.name.startswith("dfl_") else "Groningen")
                    rows.append(row)
                print(f"P3 removal diagnostic {raw.name}/{scenario}/seed={seed}", flush=True)
    df = pd.DataFrame(rows)
    cols = ["spell_f1", "spell_f1_merged", "strict_team_acc", "conditional_team_acc",
            "conditional_log_loss", "strict_burst_team_acc", "short_2s_rec"]
    summary = df.groupby(["variant", "provider", "scenario"], sort=False)[cols].mean().reset_index()
    report = dict(protocol=predeclared, training_provenance=manifest,
                  pipeline_sha256=sha256(Path(args.pipeline).read_bytes()).hexdigest(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, summary=summary.to_dict("records"),
                  invariant_dead_probability_max_diff=dead_diff,
                  cached_decode_predict_parity_cases=parity_cases,
                  uncached_feature_cache_parity_cases=uncached_parity_cases,
                  elapsed_seconds=time.monotonic() - start)
    write_json(args.out, report)
    print(summary.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
