"""Train PipelineV4 on external fixtures with fixture-group OOF preprocessing.

Each OOF classifier AND ball imputer are fitted only on that fold's training
fixtures. Training features are recomputed per fold deliberately: a global
imputer leaks held-out ball targets into ownership features. The final Stage-1
model is fitted separately. ``fit_pipeline`` supports independent benchmark splits.
"""
import argparse
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import os
import time
from typing import Callable, List, Sequence

import numpy as np

from solution import ball_tracker, datasets, external, features_v4, pipeline_v4, viterbi4
from solution.common import A, D, H, ROOT, X, Match


@dataclass
class TrainingSpec:
    label: str
    fixture: str
    fold: int
    include_dead: bool
    stride: int
    seed: int
    builder: Callable[[], Match]


def fixture_seed(fixture: str, scenario: str, seed: int = 0) -> int:
    """Stable per-fixture/scenario seed independent of Python's randomized hash."""
    key = f"pipeline-v4-v2/{int(seed)}/{fixture}/{scenario}".encode("utf-8")
    return int.from_bytes(sha256(key).digest()[:4], "little")


def lightweight_match_meta(m: Match) -> Match:
    """Retain only metadata consumed by Stage-2 feature/weight construction."""
    return Match(name=m.name, ball=np.empty((m.n, 2), dtype=np.float32),
                 pos=np.empty((0, 0, 2), dtype=np.float32),
                 spd=np.empty((0, 0), dtype=np.float32), team=m.team,
                 blocks=m.blocks, pitch=m.pitch, gt=m.gt, tags=None, meta=m.meta)


def extract_s1_features(m: Match, imputer, use_p3: bool = False):
    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    rb = ball_tracker.reconstruct_ball(m, imputer=imputer, vel=vel)
    return (features_v4.build_dead_features(m, vel=vel),
            features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=use_p3),
            features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=use_p3))


def build_training_specs(matches: Sequence[Match], inner_folds: int = 3,
                         seed: int = 0, long_bursts: bool = False,
                         stoppage_augmentation: bool = False,
                         metrica_tracking: str = "unified_v4",
                         published_degradation: bool = False) -> List[TrainingSpec]:
    """Keep every view of a fixture in one fold; never randomly split frames."""
    names = [m.name for m in matches]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate base fixture names are not independent groups")
    nfold = min(int(inner_folds), len(matches))
    if nfold < 2:
        raise ValueError("At least two fixture groups are required for Stage-2 OOF")
    fold_by_name = {name: i % nfold for i, name in enumerate(sorted(names))}
    specs = []
    for m in matches:
        source = str(m.meta.get("source", ""))
        is_metrica = "metrica" in source.lower()
        include_dead = bool(m.meta.get("has_real_dead_ball", True))
        if is_metrica:
            scenarios = ("path", "hold", "hidden")
        elif source == "sportec-dfl-v1":
            scenarios = ("natural", "atd_real", "atd_blind", "atd_carried_dead")
        else:
            scenarios = ("natural", "atd_real")
        if published_degradation and not is_metrica:
            # atd_real's dead/live dropout split has no published source; keep published-rate views only
            scenarios = tuple(dict.fromkeys(("atd_blind" if sc == "atd_real" else sc) for sc in scenarios))
        stride = 4 if include_dead else 6
        for scenario in scenarios:
            sd = fixture_seed(m.name, scenario, seed)
            if is_metrica:
                builder = lambda mr=m, sc=scenario, rs=sd: external.degrade(
                    mr, sc, seed=rs, tracking_version=metrica_tracking)
            else:
                builder = lambda mr=m, sc=scenario, rs=sd: datasets.degrade_atd(mr, scenario=sc, seed=rs)
            specs.append(TrainingSpec(f"{m.name}/{scenario}", m.name,
                                      fold_by_name[m.name], include_dead, stride, sd, builder))
        for enabled, aug in ((long_bursts, "long_burst"),
                             (stoppage_augmentation, "stoppage_motion")):
            if not enabled:
                continue
            sd = fixture_seed(m.name, aug, seed)
            def augmented(mr=m, rs=sd, kind=aug):
                view = datasets.degrade_atd(mr, scenario="atd_blind" if published_degradation else "atd_real", seed=rs)
                if kind == "long_burst":
                    return datasets.augment_long_burst_gaps(view, seed=rs + 1, prob=1.0,
                                                          gap_range=(125, 750))
                return datasets.augment_stoppage_ball(view, seed=rs + 1)
            specs.append(TrainingSpec(f"{m.name}/{aug}", m.name,
                                      fold_by_name[m.name], include_dead, stride, sd, augmented))
    return specs


def _sample_training(specs, imputer, use_p3=False, stride_scale=1):
    buckets = dict(Xd=[], yd=[], wd=[], Xt=[], yt=[], wt=[])
    for spec in specs:
        m = spec.builder()
        if m.gt is None:
            raise ValueError("Training requires labelled external fixtures")
        Xd, Xf, Xs = extract_s1_features(m, imputer, use_p3=use_p3)
        mask = np.zeros(m.n, dtype=bool)
        for a, b in m.blocks:
            mask[a:b:spec.stride * int(stride_scale)] = True
        bw = pipeline_v4.boundary_weights(m)
        dm = mask & (m.gt != X)
        if spec.include_dead and m.meta.get("has_real_dead_ball", True) and dm.any():
            buckets["Xd"].append(Xd[dm])
            buckets["yd"].append((m.gt[dm] == D).astype(np.int8))
            buckets["wd"].append(bw[dm])
        lm = mask & ((m.gt == H) | (m.gt == A))
        if lm.any():
            buckets["Xt"].extend((Xf[lm], Xs[lm]))
            buckets["yt"].extend(((m.gt[lm] == H).astype(np.int8),
                                  (m.gt[lm] == A).astype(np.int8)))
            buckets["wt"].extend((bw[lm], bw[lm]))
        del m, Xd, Xf, Xs
    if not buckets["Xd"] or not buckets["Xt"]:
        raise ValueError("Each fold requires real dead labels and live ownership labels")
    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}


def _fit_stage1(specs, imputer, mode, use_gpu, seed, use_p3, stride_scale,
                ball_free_fallback=False):
    arrays = _sample_training(specs, imputer, use_p3, stride_scale)
    dead = pipeline_v4.EnsembleBinaryClf(mode=mode, seed=seed, use_gpu=use_gpu).fit(
        arrays["Xd"], arrays["yd"], sample_weight=arrays["wd"])
    team = pipeline_v4.EnsembleBinaryClf(mode=mode, seed=seed + 1, use_gpu=use_gpu).fit(
        arrays["Xt"], arrays["yt"], sample_weight=arrays["wt"])
    bf, bf_cols = None, None
    if ball_free_fallback:
        if not use_p3:
            raise ValueError("Ball-free fallback requires --use-p3")
        bf_cols = features_v4.ball_free_team_cols(arrays["Xt"].shape[1])
        if bf_cols is None:
            raise ValueError("Missing ball-free feature columns")
        bf = pipeline_v4.EnsembleBinaryClf(mode=mode, seed=seed + 2, use_gpu=use_gpu).fit(
            arrays["Xt"][:, bf_cols], arrays["yt"], sample_weight=arrays["wt"])
    del arrays
    gc.collect()
    return pipeline_v4.PipelineV4(imputer=imputer, dead_s1=dead, team_s1=team,
                                  use_p3=use_p3, team_bf_s1=bf, bf_cols=bf_cols)


def fit_pipeline(raw_train_matches: Sequence[Match], skillcorner_ids=(), *,
                 mode: str = "lgbm", use_gpu: bool = False, use_p3: bool = False,
                 include_sc_stage2: bool = True, sc_s2_weight: float = 1.0,
                 seed: int = 0, inner_folds: int = 3, long_bursts: bool = False,
                 stoppage_augmentation: bool = False, ball_free_fallback: bool = False,
                 metrica_tracking: str = "unified_v4", stride_scale: int = 1,
                 imputer_stride: int = 6, decode_cfg=None, oof_dir=None, progress=print,
                 published_degradation: bool = False):
    """Return (pipeline, provenance), fitted only on the supplied fixtures.

    Exclude all evaluation fixtures here, including auxiliary datasets. Inner OOF
    protects Stage-2 construction, not evaluation on a supplied training fixture.
    Optional oof_dir stores external-only arrays locally, never on the console.
    """
    if int(stride_scale) < 1:
        raise ValueError("stride_scale must be positive")
    if not np.isfinite(sc_s2_weight) or sc_s2_weight < 0:
        raise ValueError("sc_s2_weight must be finite and nonnegative")
    t0 = time.time()
    matches = list(raw_train_matches) + [datasets.standardize_match(
        datasets.load_skillcorner_current(mid)) for mid in skillcorner_ids]
    allowed_sources = {"sportec-dfl-v1", "wcsf-groningen", "skillcorner-current", "metrica sample-data"}
    for m in matches:
        if m.gt is None or m.tags is not None or m.meta.get("source") not in allowed_sources:
            raise ValueError("Only labelled external fixtures without competition tags may train")
    specs = build_training_specs(matches, inner_folds, seed, long_bursts,
                                  stoppage_augmentation, metrica_tracking,
                                  published_degradation=published_degradation)
    folds = sorted({s.fold for s in specs})
    provenance = dict(training_recipe="pipeline_v4_nested_preprocessing_v2",
                      mode=mode, seed=int(seed), use_p3=bool(use_p3),
                      ball_free_fallback=bool(ball_free_fallback),
                      long_bursts=bool(long_bursts),
                      stoppage_augmentation=bool(stoppage_augmentation),
                      include_sc_stage2=bool(include_sc_stage2), sc_s2_weight=float(sc_s2_weight),
                      metrica_tracking=metrica_tracking, stride_scale=int(stride_scale),
                      published_degradation=bool(published_degradation),
                      fixtures=[dict(name=m.name, source=m.meta.get("source"),
                                     has_real_dead_ball=m.meta.get("has_real_dead_ball", True))
                                for m in matches],
                      scenarios=[dict(label=s.label, fixture=s.fixture, fold=s.fold, seed=s.seed)
                                 for s in specs], inner_folds=[])
    if oof_dir is not None:
        os.makedirs(oof_dir, exist_ok=True)
    oof_items = []
    for fold in folds:
        val_specs = [s for s in specs if s.fold == fold]
        tr_specs = [s for s in specs if s.fold != fold]
        train_names = {s.fixture for s in tr_specs}
        held_names = {s.fixture for s in val_specs}
        assert train_names.isdisjoint(held_names)
        imputer_matches = [m for m in matches if m.name in train_names]
        progress(f"[{time.time()-t0:.1f}s] OOF {fold}: {len(train_names)} train / {len(held_names)} held-out fixtures", flush=True)
        imp = ball_tracker.fit_constellation_imputer(imputer_matches, stride=imputer_stride,
                                                     seed=seed + fold)
        pipe = _fit_stage1(tr_specs, imp, mode, use_gpu, seed + 10 * fold,
                           use_p3, stride_scale, ball_free_fallback)
        for spec in val_specs:
            if not include_sc_stage2 and not spec.include_dead:
                continue
            m = spec.builder()
            pd, ph, Xd, Xf, Xs = pipe.predict_stage1_for_stacker(m)
            oof_items.append((pd, ph, Xd.copy(), Xf.copy(), Xs.copy(),
                              lightweight_match_meta(m),
                              1.0 if spec.include_dead else float(sc_s2_weight)))
            if oof_dir is not None:
                key = sha256(spec.label.encode("utf-8")).hexdigest()[:16]
                np.savez_compressed(os.path.join(oof_dir, key + ".npz"),
                                    p_dead=pd, p_home_live=ph, Xd=Xd, Xf=Xf, Xs=Xs,
                                    gt=m.gt, blocks=np.asarray(m.blocks),
                                    fixture=np.asarray(m.name), scenario=np.asarray(spec.label),
                                    fold=np.asarray(fold))
        provenance["inner_folds"].append(dict(fold=fold, stage1_train=sorted(train_names),
                                              imputer_train=sorted(train_names),
                                              stage2_oof=sorted(held_names)))
        del pipe, imp
        gc.collect()
    stacker = pipeline_v4.fit_temporal_stacker_from_oof(
        oof_items, stride=3 * int(stride_scale), seed=seed, mode=mode, use_gpu=use_gpu)
    del oof_items
    gc.collect()
    progress(f"[{time.time()-t0:.1f}s] Fitting final imputer and Stage-1 on {len(matches)} fixtures", flush=True)
    imputer = ball_tracker.fit_constellation_imputer(matches, stride=imputer_stride, seed=seed)
    pipe = _fit_stage1(specs, imputer, mode, use_gpu, seed, use_p3,
                       stride_scale, ball_free_fallback)
    pipe.stacker = stacker
    pipe.decode_cfg = dict(viterbi4.DECODE4_DEFAULTS if decode_cfg is None else decode_cfg)
    pipe.training_provenance = provenance
    if oof_dir is not None:
        with open(os.path.join(oof_dir, "provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2, sort_keys=True)
    progress(f"[{time.time()-t0:.1f}s] Pipeline fitted (not an evaluation score)", flush=True)
    return pipe, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dfl-games", nargs="+", choices=datasets.DFL_GAMES, default=list(datasets.DFL_GAMES))
    ap.add_argument("--no-groningen", action="store_true")
    ap.add_argument("--no-metrica", action="store_true")
    ap.add_argument("--sc-limit", type=int, default=20)
    ap.add_argument("--no-skillcorner-stage2", action="store_true")
    ap.add_argument("--published-degradation", action="store_true",
                    help="train only on degradations set from published match figures (no atd_real)")
    ap.add_argument("--sc-s2-weight", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mode", choices=("lgbm", "ens3", "hgb", "xgb"), default="ens3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--stride-scale", type=int, default=1)
    ap.add_argument("--use-p3", action="store_true")
    ap.add_argument("--ball-free-fallback", action="store_true")
    ap.add_argument("--long-bursts", action="store_true")
    ap.add_argument("--stoppage-augmentation", action="store_true")
    ap.add_argument("--metrica-tracking", choices=("legacy", "unified_v4"), default="unified_v4")
    ap.add_argument("--save-oof")
    ap.add_argument("--out", default=os.path.join(ROOT, "solution", "configs", "pipeline_v4_v2.pkl"),
                    help="Separate v2 artifact; never overwrite the legacy baseline implicitly")
    ap.add_argument("--decode-cfg", help="Explicit externally selected decoder; otherwise fixed DECODE4_DEFAULTS")
    args = ap.parse_args()
    if args.sc_limit < 0:
        ap.error("--sc-limit must be nonnegative")
    raw = [datasets.load_dfl(mid) for mid in args.dfl_games]
    if not args.no_groningen:
        raw.append(datasets.standardize_match(datasets.load_groningen()))
    if not args.no_metrica:
        raw.extend(datasets.standardize_match(external.load(g)) for g in external.GAMES)
    sc_ids = sorted(datasets.list_skillcorner_current())[:args.sc_limit]
    cfg = None
    if args.decode_cfg:
        with open(args.decode_cfg) as f:
            cfg = json.load(f)
    pipe, provenance = fit_pipeline(
        raw, sc_ids, mode=args.mode, use_gpu=not args.cpu, use_p3=args.use_p3,
        include_sc_stage2=not args.no_skillcorner_stage2, sc_s2_weight=args.sc_s2_weight,
        seed=args.seed, inner_folds=args.inner_folds, long_bursts=args.long_bursts,
        stoppage_augmentation=args.stoppage_augmentation,
        ball_free_fallback=args.ball_free_fallback, metrica_tracking=args.metrica_tracking,
        stride_scale=args.stride_scale, decode_cfg=cfg, oof_dir=args.save_oof,
        published_degradation=args.published_degradation)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pipe.save(args.out)
    with open(args.out + ".provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {args.out} and external-training provenance", flush=True)


if __name__ == "__main__":
    main()
