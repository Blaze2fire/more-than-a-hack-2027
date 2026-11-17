"""Independent external fixture split for the deadline review.

Training: DFL WMX/WN1/WOH/WOY only (+ explicitly requested SkillCorner).
Evaluation: DFL WPY/WQQ/WR9, Groningen and Metrica 1/2/3. These are historically
reused development fixtures, not an untouched final test. Current classifier,
imputer and stacker fitting excludes them; the inherited E014 decoder previously
used these development fixtures. No competition fixture loader is imported.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from solution import ablate_v4, ball_tracker, bench_v4, changepoint, datasets, external
from solution import features_v4, pipeline_v4, viterbi4
from solution.common import A, D, H, ROOT, Match

TRAIN_IDS = datasets.DFL_GAMES[:4]
TEST_IDS = datasets.DFL_GAMES[4:]


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    return value


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(safe_json(obj), indent=2, allow_nan=False) + "\n")


def source_hashes():
    names = ("train_v4.py", "pipeline_v4.py", "features_v4.py", "ball_tracker.py",
             "datasets.py", "external.py", "temporal.py", "viterbi4.py", "changepoint.py",
             "deadline_benchmark.py", "ownership_blend.py")
    return {name: hashlib.sha256((Path(ROOT) / "solution" / name).read_bytes()).hexdigest() for name in names}


IMPORTED_SOURCE_HASHES = source_hashes()


def validate_training_provenance(provenance, pipe):
    """Check actual checkpoint ancestry, not only a caller-written split label."""
    actual = getattr(pipe, "training_provenance", None)
    if not actual or actual.get("fixtures") != provenance.get("fixtures"):
        raise ValueError("Checkpoint and manifest disagree on training fixtures")
    expected = {"dfl_" + mid for mid in TRAIN_IDS}
    expected.update("sc_" + str(mid) for mid in provenance.get("training_skillcorner", []))
    fixtures = actual["fixtures"]
    names = {f["name"] for f in fixtures}
    if names != expected:
        raise ValueError("Unexpected fitted fixtures: " + repr(sorted(names ^ expected)))
    forbidden = {"dfl_" + mid for mid in TEST_IDS} | {"groningen", "metrica1", "metrica2", "metrica3"}
    if names & forbidden:
        raise ValueError("An evaluation fixture entered model fitting")
    for fold in actual["inner_folds"]:
        train_names = set(fold["stage1_train"])
        imputer_names = set(fold["imputer_train"])
        held = set(fold["stage2_oof"])
        if train_names != imputer_names or held & train_names or not (train_names | held) <= names:
            raise ValueError("Invalid preprocessing/model ancestry in inner OOF")
    return names


def frozen_baseline_config():
    """Use the shipped artifact's actual decoder, not a subsequently edited sidecar."""
    path = Path(ROOT) / "solution/configs/pipeline_v4.pkl"
    return dict(pipeline_v4.PipelineV4.load(str(path)).decode_cfg)


def train(args):
    from solution import train_v4
    if not 0 <= args.sc_limit <= 20:
        raise ValueError("--sc-limit must be between 0 and 20")
    start = time.monotonic()
    raw = [datasets.load_dfl(mid) for mid in TRAIN_IDS]
    if args.metrica_train:
        raw.extend(datasets.standardize_match(external.load(g)) for g in external.GAMES)
    sc = sorted(datasets.list_skillcorner_current())[:args.sc_limit]
    pipe, manifest = train_v4.fit_pipeline(
        raw, skillcorner_ids=sc, mode=args.mode, use_gpu=args.gpu,
        use_p3=args.use_p3, include_sc_stage2=not args.no_sc_stage2,
        seed=args.seed, inner_folds=args.inner_folds,
        ball_free_fallback=args.ball_free, long_bursts=args.long_bursts,
        stoppage_augmentation=args.stoppage_augmentation,
        oof_dir=args.save_oof, published_degradation=args.published_degradation,
    )
    pipe.decode_cfg = frozen_baseline_config()
    # Optional additions are fitted on exactly the same explicit training set.
    if args.boundary:
        boundary_records, xb, yb = [], [], []
        bf_cols = None
        for i, match in enumerate(raw):
            for scenario in ("natural", "atd_real"):
                m = datasets.degrade_atd(match, scenario, seed=args.seed + 811 + i)
                vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
                rb = ball_tracker.reconstruct_ball(m, imputer=pipe.imputer, vel=vel)
                xf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=args.use_p3)
                xs = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=args.use_p3)
                if args.boundary:
                    boundary_records.append((m.name, xf, xs, m.gt, m.blocks))
        if args.boundary:
            pipe.boundary_model = changepoint.fit(boundary_records, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pipe.save(args.out)
    manifest.update({
        "benchmark_protocol": "independent first-four DFL training; fixture-group inner OOF",
        "training_dfl": list(TRAIN_IDS), "training_skillcorner": sc,
        "held_out_dfl": list(TEST_IDS),
        "held_out_providers": ["groningen"] + ([] if args.metrica_train else ["metrica1", "metrica2", "metrica3"]),
        "metrica_used_for_fitting": bool(args.metrica_train), "competition_data_accessed": False,
        "source_hashes": source_hashes(), "import_source_hashes": IMPORTED_SOURCE_HASHES,
        "args": vars(args),
        "model_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - start,
        "caveat": "External fixtures have prior development exposure; no untouched-test claim.",
    })
    write_json(args.out + ".manifest.json", manifest)
    print("Saved independent model and provenance:", args.out, flush=True)


def external_cases(metrica_games):
    for mid in TEST_IDS:
        yield datasets.load_dfl(mid)
    yield datasets.load_groningen()
    for game in metrica_games:
        m = datasets.standardize_match(external.load(f"metrica{game}"))
        # Score both event-raw and the original controlled-touch derivation;
        # never silently call already-folded Metrica labels "raw".
        yield dataclasses.replace(m, gt=np.asarray(m.meta["gt_raw"], dtype=np.int8),
                                  meta=dict(m.meta, gt_merged_reference=m.gt.copy()))


def coherent_carried_dead(raw, seed):
    """Adversarial external stoppages with a moving ball next to a tracked player."""
    m = datasets.degrade_atd(raw, "atd_carried_dead", seed)
    ball = m.ball.copy()
    finite = np.isfinite(m.pos).all(axis=2)
    slot = np.argmax(finite, axis=1)
    carrier = m.pos[np.arange(m.n), slot]
    dead = (m.gt == D) & finite.any(axis=1)
    ball[dead] = carrier[dead] + np.array([0.15, -0.1], dtype=np.float32)
    return dataclasses.replace(m, ball=ball, meta=dict(m.meta, scenario="moving_dead"))


def invariant_probe(pipe, m):
    # Small block-contained public-external slice; no scoring or tuning on this probe.
    lo, hi = m.blocks[0]
    hi = min(hi, lo + 1200)
    tiny = dataclasses.replace(m, ball=m.ball[lo:hi].copy(), pos=m.pos[lo:hi],
                               spd=m.spd[lo:hi], blocks=[(0, hi - lo)],
                               gt=m.gt[lo:hi], meta={"source": m.meta.get("source")})
    result = ablate_v4.check_invariants(pipe, tiny)
    if result["dead_leak"] != 0 or result["swap_diff"] != 0:
        raise AssertionError("External pipeline invariant failed: " + repr(result))
    return result


def decoder_variants(base):
    fixed = dict(base)
    fixed["c_res_same"], fixed["c_res_flip"] = sorted((base["c_res_same"], base["c_res_flip"]))
    gentle = dict(fixed, min_dead=25, min_live_blip=10, min_turn=0)
    return {
        "baseline": dict(base),
        "restart_order": fixed,
        "relaxed_blips": gentle,
        "global_change": dict(gentle, c_sw_gain=0.55 * gentle["c_team"]),
        "two_pass": dict(fixed, ownership_pass=True, ownership_cost_scale=0.5),
        "two_pass_change": dict(fixed, ownership_pass=True, ownership_cost_scale=0.5,
                                c_sw_gain=0.55 * fixed["c_team"]),
        "frozen_dead": dict(fixed, freeze_dead_from_gbdt=True),
    }


def summarize(rows):
    if not rows:
        return []
    df = pd.DataFrame(rows)
    cols = ["spell_f1", "spell_f1_merged", "macro_f1", "alive_dead_f1", "team_acc", "short_2s_rec", "burst_team_acc"]
    return df.groupby(["variant", "provider", "scenario"], sort=False)[cols].mean().reset_index().to_dict("records")


def cached_neural_predictions(models, manifest, m, scenario, seed, cache_dir):
    """Public-external cache bound to both checkpoint and exact degraded inputs."""
    from solution import neural_v2
    digest = hashlib.sha256()
    for value in (m.ball, m.pos, m.team, np.asarray(m.blocks)):
        digest.update(np.ascontiguousarray(value).tobytes())
    input_hash = digest.hexdigest()
    checkpoint_hash = manifest["model_sha256"]
    folder = Path(cache_dir) / checkpoint_hash[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{m.name}__{scenario}__s{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["input_sha256"]) == input_hash and str(saved["model_sha256"]) == checkpoint_hash:
                return saved["p_home"].copy(), saved["p_switch"].copy()
    p, q = neural_v2.predict_match(models, m)
    np.savez_compressed(path, p_home=p, p_switch=q, input_sha256=input_hash, model_sha256=checkpoint_hash)
    return p, q


def evaluate(args):
    start = time.monotonic()
    manifest_path = args.pipeline + ".manifest.json"
    if not os.path.isfile(manifest_path):
        raise ValueError("Independent evaluation requires a training provenance manifest")
    provenance = json.loads(Path(manifest_path).read_text())
    if provenance.get("metrica_used_for_fitting") and args.metrica:
        raise ValueError("Metrica-trained model: evaluate with --metrica \"\" (DFL + Groningen only)")
    if provenance.get("training_dfl") != list(TRAIN_IDS):
        raise ValueError("This comparison requires exactly the fixed first-four DFL split and no Metrica fitting")
    actual_hash = hashlib.sha256(Path(args.pipeline).read_bytes()).hexdigest()
    if actual_hash != provenance["model_sha256"]:
        raise ValueError("Model changed since its provenance manifest was written")
    pipe = pipeline_v4.PipelineV4.load(args.pipeline)
    training_names = validate_training_provenance(provenance, pipe)
    models = None
    neural_manifest = None
    if args.neural:
        model_path = Path(args.neural)
        neural_manifest_path = model_path.with_name(model_path.name.replace("models_", "manifest_").replace(".pkl", ".json"))
        neural_manifest = json.loads(neural_manifest_path.read_text())
        if neural_manifest.get("train_keys") != ["dfl0", "dfl1", "dfl2", "dfl3"]:
            raise ValueError("Neural model must use the same first-four DFL split")
        if neural_manifest.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("Neural checkpoint hash mismatch")
        with open(args.neural, "rb") as handle:
            models = pickle.load(handle)
    variants = decoder_variants(pipe.decode_cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, invariants = [], {}
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = args.scenarios.split(",")
    for raw in external_cases([int(g) for g in args.metrica.split(",") if g]):
        provider = "DFL" if raw.name.startswith("dfl_") else ("Groningen" if raw.name == "groningen" else "Metrica")
        if raw.name in training_names:
            raise ValueError("Train/test fixture overlap: " + raw.name)
        for scenario in scenarios:
            for seed in (seeds[:1] if scenario in ("natural", "gap-stress-burst-01") else seeds):
                if args.skip_saved and (out / f"{raw.name}__{scenario}__s{seed}__nn1_change.npz").exists():
                    continue
                if scenario.startswith("gap-stress"):
                    m = ablate_v4.make_elvern_burst_match(datasets.degrade_atd(raw, "natural", seed), raw.name.replace("dfl_", ""), scenario)
                elif scenario == "moving_dead":
                    m = coherent_carried_dead(raw, seed)
                else:
                    m = datasets.degrade_atd(raw, scenario, seed)
                rb, vel, pd1, pha1, bf, nn, switch, lp, sw = pipe.predict_components(m)
                if raw.name not in invariants:
                    invariants[raw.name] = invariant_probe(pipe, m)
                gt_merged = m.meta.get("gt_merged_reference")
                if gt_merged is None:
                    gt_merged = bench_v4.merge_uncontrolled_gt(m)
                gap = bench_v4.gap_lengths(np.isfinite(m.ball[:, 0]), m.blocks)
                neural_own, neural_sw = None, None
                if models:
                    neural_own, neural_sw = cached_neural_predictions(models, neural_manifest, m, scenario, seed, args.neural_cache)
                mixtures = [("gbdt", lp, sw)]
                fixed_dead_reference = viterbi4.decode4(lp, m.blocks, pipe.decode_cfg)
                if neural_own is not None:
                    from solution.ownership_blend import blend_logp
                    for weight in (0.5, 0.7, 1.0):
                        blended = blend_logp(lp, neural_own, weight)
                        mixtures.append((f"nn{weight:g}_ownership", blended, sw))
                        mixtures.append((f"nn{weight:g}_change", blended, neural_sw))
                        assert np.array_equal(lp[:, D], blended[:, D]), "Neural blend altered dead probability"
                for mix_name, mixture, signal in mixtures:
                    # Save only explicitly public external predictions; never competition data.
                    stem = f"{m.name}__{scenario}__s{seed}__{mix_name}"
                    if args.save_predictions:
                        np.savez_compressed(out / (stem + ".npz"), lp=mixture, sw_sig=signal if signal is not None else [],
                                            gt=m.gt, gt_merged=gt_merged, blocks=np.asarray(m.blocks), rb_gap=gap)
                    for decoder_name, cfg in variants.items():
                        if cfg.get("freeze_dead_from_gbdt"):
                            pred = viterbi4.ownership_pass(cfg["w"] * mixture, fixed_dead_reference,
                                                         m.blocks, cfg["c_team"] * 0.5,
                                                         sw_signal=signal, c_sw_gain=cfg.get("c_sw_gain", 0) * 0.5)
                            assert np.array_equal(pred == D, fixed_dead_reference == D)
                        else:
                            pred = viterbi4.decode4(mixture, m.blocks, cfg, sw_signal=signal)
                        row = bench_v4.score_match(m.name, scenario, pred, m.gt, gt_merged)
                        row.update(ablate_v4.compute_short_and_trans_metrics(pred, m.gt, m.blocks))
                        live_burst = (gap > 25) & ((m.gt == H) | (m.gt == A)) & (pred != D)
                        row["burst_team_acc"] = float(np.mean(pred[live_burst] == m.gt[live_burst])) if live_burst.any() else None
                        gt_live = (m.gt == H) | (m.gt == A)
                        team_pred = np.argmax(mixture[:, :2], axis=1)
                        row["conditional_team_acc_all_live"] = float(np.mean(team_pred[gt_live] == m.gt[gt_live]))
                        burst_all = gt_live & (gap > 25)
                        row["burst_strict_team_acc"] = float(np.mean(pred[burst_all] == m.gt[burst_all])) if burst_all.any() else None
                        row["burst_live_coverage"] = float(np.mean(pred[burst_all] != D)) if burst_all.any() else None
                        row.update(provider=provider, seed=seed, variant=mix_name + "/" + decoder_name)
                        rows.append(row)
                print(f"Evaluated external {m.name}/{scenario}/seed={seed}", flush=True)
        write_json(out / "results.json", {"protocol": provenance, "neural_provenance": neural_manifest,
                   "decoder_variants": variants, "invariants": invariants,
                   "rows": rows, "summary": summarize(rows), "elapsed_seconds": time.monotonic() - start})
    summary = summarize(rows)
    print(pd.DataFrame(summary)[["variant", "provider", "scenario", "spell_f1", "macro_f1"]].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--out", required=True)
    training.add_argument("--mode", choices=("lgbm", "xgb", "ens3"), default="lgbm")
    training.add_argument("--sc-limit", type=int, default=0)
    training.add_argument("--no-sc-stage2", action="store_true")
    training.add_argument("--use-p3", action="store_true")
    training.add_argument("--ball-free", action="store_true")
    training.add_argument("--boundary", action="store_true")
    training.add_argument("--long-bursts", action="store_true")
    training.add_argument("--stoppage-augmentation", action="store_true")
    training.add_argument("--save-oof")
    training.add_argument("--seed", type=int, default=1701)
    training.add_argument("--inner-folds", type=int, default=3)
    training.add_argument("--gpu", action="store_true", help="XGBoost CUDA + CatBoost GPU (default CPU)")
    training.add_argument("--metrica-train", action="store_true", help="also fit on Metrica 1-3 (then evaluate without Metrica)")
    training.add_argument("--published-degradation", action="store_true", help="no atd_real training views")
    testing = sub.add_parser("evaluate")
    testing.add_argument("--pipeline", required=True)
    testing.add_argument("--neural")
    testing.add_argument("--out-dir", required=True)
    testing.add_argument("--seeds", default="911,2027")
    testing.add_argument("--scenarios", default="natural,atd_real,atd_blind,atd_carried_dead,gap-stress-burst-01")
    testing.add_argument("--metrica", default="1,2,3")
    testing.add_argument("--save-predictions", action="store_true")
    testing.add_argument("--skip-saved", action="store_true", help="resume: skip cases whose predictions are already saved")
    testing.add_argument("--neural-cache", default="solution/out/deadline_neural_eval_cache")
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
