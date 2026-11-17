"""Isolate SkillCorner Stage-2 ownership supervision using a frozen DFL4 Stage-1.

Reuses a completed deadline baseline and its external inner-OOF arrays. No
Stage-1/imputer refit, no Metrica fitting, and the dead heads stay identical.
The SkillCorner fixtures are wholly held out from Stage-1/imputer fitting, so
their Stage-2 input probabilities are out-of-training predictions too.
"""
import argparse
import dataclasses
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from solution import bench_v4, datasets, pipeline_v4, temporal, train_v4, viterbi4
from solution.common import A, D, H, Match


TRAIN_IDS = list(datasets.DFL_GAMES[:4])
TEST_IDS = list(datasets.DFL_GAMES[4:])


def write_json(path, obj):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, (np.integer, np.floating)):
            return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(convert(obj), indent=2, allow_nan=False) + "\n")


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(model_path):
    manifest = json.loads(Path(str(model_path) + ".manifest.json").read_text())
    if manifest.get("training_dfl") != TRAIN_IDS or manifest.get("metrica_used_for_fitting") is not False:
        raise ValueError("Requires the independent first-four DFL baseline, no Metrica fitting")
    if manifest.get("training_skillcorner"):
        raise ValueError("This isolated Stage-2 addition requires no baseline SkillCorner fitting")
    if _hash(model_path) != manifest["model_sha256"]:
        raise ValueError("Baseline model hash mismatch")
    pipe = pipeline_v4.PipelineV4.load(str(model_path))
    actual = getattr(pipe, "training_provenance", None)
    expected_names = {"dfl_" + mid for mid in TRAIN_IDS}
    if actual is None or {x["name"] for x in actual["fixtures"]} != expected_names:
        raise ValueError("Model's own provenance does not match DFL4")
    if actual.get("stage2_skillcorner_added"):
        raise ValueError("Baseline already contains added SkillCorner Stage-2 supervision")
    if {x["source"] for x in actual["fixtures"]} != {"sportec-dfl-v1"}:
        raise ValueError("Unexpected fitted provider")
    for fold in actual["inner_folds"]:
        hold = set(fold["stage2_oof"])
        if hold.intersection(fold["imputer_train"]) or hold.intersection(fold["stage1_train"]):
            raise ValueError("Baseline inner-OOF leakage")
    if pipe.stacker is None:
        raise ValueError("Baseline must include Stage-2")
    return pipe, manifest


def load_external_oof(directory, provenance):
    directory = Path(directory)
    saved = json.loads((directory / "provenance.json").read_text())
    for field in ("fixtures", "scenarios", "inner_folds", "seed", "training_recipe"):
        if saved[field] != provenance[field]:
            raise ValueError("OOF provenance mismatch: " + field)
    raw = {"dfl_" + mid: datasets.load_dfl(mid) for mid in TRAIN_IDS}
    items, hashes = [], {}
    for fold in sorted({s["fold"] for s in saved["scenarios"]}):
        for spec in saved["scenarios"]:
            if spec["fold"] != fold:
                continue
            key = sha256(spec["label"].encode("utf-8")).hexdigest()[:16] + ".npz"
            path = directory / key
            hashes[key] = _hash(path)
            with np.load(path, allow_pickle=False) as z:
                name = str(z["fixture"])
                if name not in raw or str(z["scenario"]) != spec["label"]:
                    raise ValueError("Unregistered fixture/scenario in OOF cache")
                if int(z["fold"]) != int(fold):
                    raise ValueError("OOF fold does not match its provenance")
                n = len(z["gt"])
                blocks = [tuple(map(int, x)) for x in z["blocks"]]
                stub = Match(name=name, ball=np.empty((n, 2), np.float32),
                             pos=np.empty((0, 0, 2), np.float32), spd=np.empty((0, 0), np.float32),
                             team=raw[name].team, blocks=blocks, pitch=raw[name].pitch,
                             gt=z["gt"].copy(), meta=dict(raw[name].meta))
                items.append((z["p_dead"].copy(), z["p_home_live"].copy(), None,
                              z["Xf"].copy(), z["Xs"].copy(), stub, 1.0))
    return items, hashes


def fit_ownership_stacker(items, mode, seed):
    """Fit only live ownership; never even construct dead training examples."""
    features, labels, weights = [], [], []
    for pd, ph, _, xf, xs, m, weight in items:
        mask = np.zeros(m.n, dtype=bool)
        for a, b in m.blocks:
            mask[a:b:3] = True
        mask &= (m.gt == H) | (m.gt == A)
        if not mask.any():
            continue
        bw = pipeline_v4.boundary_weights(m)[mask] * float(weight)
        for swap in (False, True):
            z = temporal.build_team_seq_features(ph, pd, xf, m.blocks, swap=swap, Xs_base=xs)
            features.append(z[mask])
            labels.append((m.gt[mask] == (A if swap else H)).astype(np.int8))
            weights.append(bw)
    x_all, y_all, w_all = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    del features, labels, weights
    gc.collect()
    return pipeline_v4.EnsembleBinaryClf(mode=mode, seed=seed + 1, use_gpu=False).fit(
        x_all, y_all, sample_weight=w_all)


def fit(args):
    start = time.monotonic()
    if not np.isfinite(args.sc_weight) or args.sc_weight < 0:
        raise ValueError("SkillCorner ownership weight must be finite and nonnegative")
    pipe, base_manifest = load_baseline(args.baseline)
    provenance = pipe.training_provenance
    if provenance.get("stride_scale", 1) != 1:
        raise ValueError("Current paired recipe requires baseline stride_scale=1")
    items, oof_hashes = load_external_oof(args.oof_dir, provenance)
    if args.sc_limit < 1:
        raise ValueError("At least one SkillCorner fixture is required")
    sc_ids = sorted(datasets.list_skillcorner_current())[:args.sc_limit]
    for mid in sc_ids:
        raw = datasets.standardize_match(datasets.load_skillcorner_current(mid))
        if raw.meta.get("has_real_dead_ball") is not False:
            raise ValueError("SkillCorner must have no supervised dead labels")
        for scenario in ("natural", "atd_real"):
            sd = train_v4.fixture_seed(raw.name, scenario, int(provenance["seed"]))
            m = datasets.degrade_atd(raw, scenario, sd)
            pd, ph, xd, xf, xs = pipe.predict_stage1_for_stacker(m)
            items.append((pd, ph, None, xf.copy(), xs.copy(),
                          train_v4.lightweight_match_meta(m), args.sc_weight))
            print(f"Prepared held-out-from-Stage1 SkillCorner {mid}/{scenario}", flush=True)
    owner = fit_ownership_stacker(items, provenance["mode"], int(provenance["seed"]))
    candidate = dataclasses.replace(pipe, stacker=temporal.TemporalStacker(
        dead_s2=pipe.stacker.dead_s2, team_s2=owner))
    candidate.training_provenance = dict(provenance, stage2_skillcorner_added=sc_ids,
                                         stage2_skillcorner_weight=float(args.sc_weight))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    candidate.save(args.out)
    manifest = dict(protocol="paired SkillCorner Stage-2-only addition to frozen DFL4",
                    baseline_model=str(Path(args.baseline).resolve()),
                    baseline_sha256=_hash(args.baseline), model_sha256=_hash(args.out),
                    training_dfl=TRAIN_IDS, stage1_skillcorner=[], stage2_skillcorner=sc_ids,
                    skillcorner_weight=float(args.sc_weight), metrica_used_for_fitting=False,
                    common_dead_heads=True, common_stage1=True, oof_hashes=oof_hashes,
                    training_provenance=candidate.training_provenance,
                    baseline_manifest=base_manifest, elapsed_seconds=time.monotonic() - start,
                    source_sha256=_hash(__file__),
                    confirmation_guard=dict(mean_accuracy_drop_max=0.01,
                                            worst_match_accuracy_drop_max=0.03,
                                            averaging="equal mean over natural and atd_real, then equal mean over fixtures",
                                            role="descriptive non-regression guard, not an untouched-test certificate"),
                    caveat="SC Stage-2 rows use all-four-DFL Stage-1; DFL rows use inner-fold Stage-1. "
                           "Neither Stage-1 ancestor trains on its input fixture. Reused external development split.")
    write_json(str(args.out) + ".manifest.json", manifest)
    print("Saved isolated Stage-2 candidate", args.out, flush=True)


def evaluate(args):
    start = time.monotonic()
    if args.sc_eval_limit < 0:
        raise ValueError("--sc-eval-limit must be nonnegative")
    candidate_manifest = json.loads(Path(str(args.candidate) + ".manifest.json").read_text())
    base_path = candidate_manifest["baseline_model"]
    base, _ = load_baseline(base_path)
    if _hash(base_path) != candidate_manifest["baseline_sha256"] or _hash(args.candidate) != candidate_manifest["model_sha256"]:
        raise ValueError("Paired artifact hash mismatch")
    candidate = pipeline_v4.PipelineV4.load(args.candidate)
    rows = []
    dead_diff = 0.0
    cases = [datasets.load_dfl(mid) for mid in TEST_IDS] + [datasets.load_groningen()]
    for raw in cases:
        for scenario in ("natural", "atd_real"):
            for sd in ([911] if scenario == "natural" else [911, 2027]):
                m = datasets.degrade_atd(raw, scenario, sd)
                p_dead, ph, xd, xf, xs = base.predict_stage1_for_stacker(m)
                merged = bench_v4.merge_uncontrolled_gt(m)
                for name, pipe in (("without_sc_stage2", base), ("with_sc_stage2", candidate)):
                    lp = pipe.stacker.predict_log_proba(p_dead, ph, xd, xf, m.blocks, Xs_base=xs)
                    if name == "without_sc_stage2":
                        reference = lp[:, D].copy()
                    else:
                        dead_diff = max(dead_diff, float(np.max(np.abs(reference - lp[:, D]))))
                        np.testing.assert_array_equal(reference, lp[:, D])
                    pred = viterbi4.decode4(lp, m.blocks, base.decode_cfg)
                    row = bench_v4.score_match(m.name, scenario, pred, m.gt, merged)
                    live = (m.gt == H) | (m.gt == A)
                    row["strict_team_acc"] = float(np.mean(pred[live] == m.gt[live]))
                    conditional = np.argmax(lp[:, :2], axis=1)
                    row["conditional_team_acc"] = float(np.mean(conditional[live] == m.gt[live]))
                    row.update(variant=name, seed=sd,
                               provider="DFL" if m.name.startswith("dfl_") else "Groningen")
                    rows.append(row)
                print(f"Paired external evaluation {raw.name}/{scenario}/seed={sd}", flush=True)
    df = pd.DataFrame(rows)
    cols = ["spell_f1", "spell_f1_merged", "macro_f1", "alive_dead_f1", "team_acc",
            "strict_team_acc", "conditional_team_acc"]
    summary = df.groupby(["variant", "provider", "scenario"], sort=False)[cols].mean().reset_index()
    delta = df.pivot(index=["game", "provider", "scenario", "seed"], columns="variant", values="spell_f1")
    delta["delta"] = delta["with_sc_stage2"] - delta["without_sc_stage2"]
    sc_rows = []
    if args.sc_eval_limit:
        ids = sorted(datasets.list_skillcorner_current())[-args.sc_eval_limit:]
        if set(ids).intersection(candidate_manifest["stage2_skillcorner"]):
            raise ValueError("SkillCorner train/evaluation fixture overlap")
        for mid in ids:
            raw = datasets.standardize_match(datasets.load_skillcorner_current(mid))
            for scenario in ("natural", "atd_real"):
                m = datasets.degrade_atd(raw, scenario, 911)
                p_dead, ph, xd, xf, xs = base.predict_stage1_for_stacker(m)
                live = (m.gt == H) | (m.gt == A)
                for name, pipe in (("without_sc_stage2", base), ("with_sc_stage2", candidate)):
                    lp = pipe.stacker.predict_log_proba(p_dead, ph, xd, xf, m.blocks, Xs_base=xs)
                    conditional = np.argmax(lp[:, :2], axis=1)
                    both_classes = [float(np.mean(conditional[m.gt == code] == code))
                                    for code in (H, A) if np.any(m.gt == code)]
                    log_cond = lp[:, :2].astype(np.float64)
                    log_cond -= np.logaddexp(log_cond[:, 0], log_cond[:, 1])[:, None]
                    loss = -log_cond[np.flatnonzero(live), m.gt[live]]
                    sc_rows.append(dict(game=m.name, scenario=scenario, seed=911, variant=name,
                                        n_labelled=int(live.sum()),
                                        conditional_team_acc=float(np.mean(conditional[live] == m.gt[live])),
                                        balanced_accuracy=float(np.mean(both_classes)),
                                        conditional_log_loss=float(np.mean(loss))))
                print(f"Ownership-only SkillCorner confirmation {mid}/{scenario}", flush=True)
    guard = None
    if sc_rows:
        sframe = pd.DataFrame(sc_rows)
        paired = sframe.groupby(["game", "variant"])["conditional_team_acc"].mean().unstack("variant")
        drops = paired["without_sc_stage2"] - paired["with_sc_stage2"]
        guard = dict(thresholds=candidate_manifest["confirmation_guard"],
                     n_fixtures=int(len(drops)), mean_accuracy_drop=float(drops.mean()),
                     worst_match_accuracy_drop=float(drops.max()),
                     per_fixture_accuracy_drop={str(k): float(v) for k, v in drops.items()},
                     pass_mean=bool(drops.mean() <= 0.01),
                     pass_worst_match=bool(drops.max() <= 0.03),
                     passed=bool(drops.mean() <= 0.01 and drops.max() <= 0.03))
    report = dict(protocol=candidate_manifest, rows=rows, summary=summary.to_dict("records"),
                  paired_deltas=delta.reset_index().to_dict("records"),
                  skillcorner_ownership_only=sc_rows,
                  skillcorner_descriptive_guard=guard,
                  invariant_dead_probability_max_diff=dead_diff,
                  elapsed_seconds=time.monotonic() - start)
    write_json(args.out, report)
    print(summary.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--baseline", required=True)
    train.add_argument("--oof-dir", required=True)
    train.add_argument("--sc-limit", type=int, default=2)
    train.add_argument("--sc-weight", type=float, default=1.0)
    train.add_argument("--out", required=True)
    test = sub.add_parser("evaluate")
    test.add_argument("--candidate", required=True)
    test.add_argument("--out", required=True)
    test.add_argument("--sc-eval-limit", type=int, default=0,
                      help="Optional last-N SC fixture confirmation; ownership-only metrics")
    args = ap.parse_args()
    fit(args) if args.command == "fit" else evaluate(args)


if __name__ == "__main__":
    main()
