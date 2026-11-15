"""Reproduce final PipelineV4 model weights (`solution/configs/pipeline_v4.pkl`).

Trains deterministically (`seed=0`) on all 31 external fixtures:
  - 7 Sportec DFL v1 matches (natural, atd_real, atd_blind, atd_carried_dead; dead + team)
  - 1 WCSF Groningen match (natural, atd_real; dead + team)
  - 3 Metrica Sample Games 1-3 (clean standardize_match -> external.degrade path/hold/hidden; dead + team)
  - 20 SkillCorner Current matches (natural + atd_real; team ONLY, has_real_dead_ball=False)

Run from repository root:
  .venv/bin/python -m solution.train_v4
"""
import argparse
import dataclasses
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from solution import ball_tracker, datasets, external, features_v4, pipeline_v4
from solution.common import A, D, H, ROOT, X, Match


def lightweight_match_meta(m: Match) -> Match:
    """Keep only fields needed by fit_temporal_stacker_from_oof (no full pos/ball copy)."""
    return Match(
        name=m.name,
        ball=np.empty((m.n, 2), dtype=np.float32),
        pos=np.empty((0, 0, 2), dtype=np.float32),
        spd=np.empty((0, 0), dtype=np.float32),
        team=m.team,
        blocks=m.blocks,
        pitch=m.pitch,
        gt=m.gt,
        tags=None,
        meta=m.meta,
    )


def extract_s1_features(m: Match, imputer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    rb = ball_tracker.reconstruct_ball(m, imputer=imputer, vel=vel)
    Xd = features_v4.build_dead_features(m, vel=vel)
    Xt_f = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False)
    Xt_s = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True)
    return Xd, Xt_f, Xt_s


def build_training_specs(
    raw_dfl: List[Match],
    gron_clean: Match,
    met_clean: List[Match],
    include_metrica: bool = True,
):
    """Build deterministic list of match specs (label, fold_id, include_dead, stride, builder_fn)."""
    specs = []

    for k, m in enumerate(raw_dfl):
        fold = k % 3
        for si, sc in enumerate(("natural", "atd_real", "atd_blind", "atd_carried_dead")):
            seed = 1000 + 10 * k + si
            specs.append((
                f"{m.name}/{sc}", fold, True, 4,
                lambda mr=m, s=sc, sd=seed: datasets.degrade_atd(mr, scenario=s, seed=sd),
            ))

    for si, sc in enumerate(("natural", "atd_real")):
        seed = 2000 + si
        specs.append((
            f"gron/{sc}", 0, True, 3,
            lambda mr=gron_clean, s=sc, sd=seed: datasets.degrade_atd(mr, scenario=s, seed=sd),
        ))

    if include_metrica:
        for k, m in enumerate(met_clean):
            fold = k % 3
            for sc in ("path", "hold", "hidden"):
                specs.append((
                    f"{m.name}/{sc}", fold, True, 4,
                    lambda mr=m, s=sc: external.degrade(mr, s, seed=0),
                ))

    sc_mids = sorted(datasets.list_skillcorner_current())
    for i, mid in enumerate(sc_mids):
        fold = i % 3
        specs.append((
            f"sc_{mid}/natural", fold, False, 6,
            lambda md=mid: datasets.standardize_match(datasets.load_skillcorner_current(md)),
        ))
        specs.append((
            f"sc_{mid}/atd_real", fold, False, 6,
            lambda md=mid, sd=3000 + i: datasets.degrade_atd(
                datasets.standardize_match(datasets.load_skillcorner_current(md)),
                scenario="atd_real",
                seed=sd,
            ),
        ))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-metrica", action="store_true", help="Exclude Metrica 1-3 (CC BY 4.0 / MIT only)")
    ap.add_argument("--cpu", action="store_true", help="Disable CUDA/GPU backends for XGBoost/CatBoost")
    ap.add_argument("--out", default=os.path.join(ROOT, "solution", "configs", "pipeline_v4.pkl"))
    ap.add_argument("--decode-cfg", default=os.path.join(ROOT, "solution", "configs", "viterbi4.json"))
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(0)
    use_gpu = not args.cpu

    raw_dfl = datasets.load_all_dfl()
    gron_clean = datasets.standardize_match(datasets.load_groningen())
    met_clean = [] if args.no_metrica else [datasets.standardize_match(external.load(g)) for g in external.GAMES]

    print(f"[{time.time()-t0:.1f}s] Fitting BallConstellationModel...", flush=True)
    imputer = ball_tracker.fit_constellation_imputer(
        list(raw_dfl) + [gron_clean] + list(met_clean), stride=6, seed=0
    )

    specs = build_training_specs(raw_dfl, gron_clean, met_clean, include_metrica=not args.no_metrica)
    print(f"[{time.time()-t0:.1f}s] Extracting Stage-1 sampled features across {len(specs)} scenarios...", flush=True)

    fold_buckets: Dict[int, dict] = {
        f: dict(Xd=[], yd=[], wd=[], Xt=[], yt=[], wt=[]) for f in (0, 1, 2)
    }

    for idx, (label, fold, include_dead, stride, builder) in enumerate(specs):
        m = builder()
        gt = m.gt
        assert gt is not None
        Xd, Xt_f, Xt_s = extract_s1_features(m, imputer)

        mask = np.zeros(m.n, dtype=bool)
        for b0, b1 in m.blocks:
            mask[b0:b1:stride] = True
        bw = pipeline_v4.boundary_weights(m)

        if include_dead and m.meta.get("has_real_dead_ball", True):
            dm = mask & (gt != X)
            if dm.any():
                fold_buckets[fold]["Xd"].append(Xd[dm])
                fold_buckets[fold]["yd"].append((gt[dm] == D).astype(np.int8))
                fold_buckets[fold]["wd"].append(bw[dm])

        lm = mask & ((gt == H) | (gt == A))
        if lm.any():
            fold_buckets[fold]["Xt"].append(Xt_f[lm])
            fold_buckets[fold]["yt"].append((gt[lm] == H).astype(np.int8))
            fold_buckets[fold]["wt"].append(bw[lm])
            fold_buckets[fold]["Xt"].append(Xt_s[lm])
            fold_buckets[fold]["yt"].append((gt[lm] == A).astype(np.int8))
            fold_buckets[fold]["wt"].append(bw[lm])

        del m, Xd, Xt_f, Xt_s
        if (idx + 1) % 10 == 0 or idx + 1 == len(specs):
            gc.collect()
            print(f"[{time.time()-t0:.1f}s] Sampled {idx+1}/{len(specs)} ({label})", flush=True)

    fold_arrays: Dict[int, dict] = {}
    for f in (0, 1, 2):
        b = fold_buckets[f]
        fold_arrays[f] = dict(
            Xd=np.vstack(b["Xd"]),
            yd=np.concatenate(b["yd"]),
            wd=np.concatenate(b["wd"]),
            Xt=np.vstack(b["Xt"]),
            yt=np.concatenate(b["yt"]),
            wt=np.concatenate(b["wt"]),
        )
    del fold_buckets
    gc.collect()

    oof_items = []
    for val_fold in (0, 1, 2):
        tf = time.time()
        Xd_tr = np.vstack([fold_arrays[f]["Xd"] for f in (0, 1, 2) if f != val_fold])
        yd_tr = np.concatenate([fold_arrays[f]["yd"] for f in (0, 1, 2) if f != val_fold])
        wd_tr = np.concatenate([fold_arrays[f]["wd"] for f in (0, 1, 2) if f != val_fold])
        Xt_tr = np.vstack([fold_arrays[f]["Xt"] for f in (0, 1, 2) if f != val_fold])
        yt_tr = np.concatenate([fold_arrays[f]["yt"] for f in (0, 1, 2) if f != val_fold])
        wt_tr = np.concatenate([fold_arrays[f]["wt"] for f in (0, 1, 2) if f != val_fold])

        d1_f = pipeline_v4.EnsembleBinaryClf(mode="ens3", seed=val_fold, use_gpu=use_gpu).fit(
            Xd_tr, yd_tr, sample_weight=wd_tr
        )
        t1_f = pipeline_v4.EnsembleBinaryClf(mode="ens3", seed=10 + val_fold, use_gpu=use_gpu).fit(
            Xt_tr, yt_tr, sample_weight=wt_tr
        )
        del Xd_tr, yd_tr, wd_tr, Xt_tr, yt_tr, wt_tr
        gc.collect()

        val_specs = [sp for sp in specs if sp[1] == val_fold and sp[2]]
        for _, _, _, _, builder in val_specs:
            m = builder()
            Xd, Xt_f, Xt_s = extract_s1_features(m, imputer)
            p_d1 = np.clip(d1_f.predict_proba(Xd)[:, 1], 1e-4, 1.0 - 1e-4).astype(np.float32)
            ph = t1_f.predict_proba(Xt_f)[:, 1].astype(np.float64)
            pa = t1_f.predict_proba(Xt_s)[:, 1].astype(np.float64)
            d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
            d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
            p_ha1 = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)
            oof_items.append((
                p_d1,
                p_ha1,
                Xd[:, :40].astype(np.float32, copy=True),
                Xt_f[:, :65].astype(np.float32, copy=True),
                Xt_s[:, :65].astype(np.float32, copy=True),
                lightweight_match_meta(m),
            ))
            del m, Xd, Xt_f, Xt_s
        del d1_f, t1_f
        gc.collect()
        print(f"[{time.time()-t0:.1f}s] Group OOF fold {val_fold} ({len(val_specs)} matches) in {time.time()-tf:.1f}s.", flush=True)

    stacker = pipeline_v4.fit_temporal_stacker_from_oof(
        oof_items, stride=3, seed=0, mode="ens3", use_gpu=use_gpu
    )
    del oof_items
    gc.collect()

    Xd_all = np.vstack([fold_arrays[f]["Xd"] for f in (0, 1, 2)])
    yd_all = np.concatenate([fold_arrays[f]["yd"] for f in (0, 1, 2)])
    wd_all = np.concatenate([fold_arrays[f]["wd"] for f in (0, 1, 2)])
    for f in (0, 1, 2):
        del fold_arrays[f]["Xd"], fold_arrays[f]["yd"], fold_arrays[f]["wd"]
    gc.collect()

    dead_s1 = pipeline_v4.EnsembleBinaryClf(mode="ens3", seed=0, use_gpu=use_gpu).fit(
        Xd_all, yd_all, sample_weight=wd_all
    )
    del Xd_all, yd_all, wd_all
    gc.collect()

    Xt_all = np.vstack([fold_arrays[f]["Xt"] for f in (0, 1, 2)])
    yt_all = np.concatenate([fold_arrays[f]["yt"] for f in (0, 1, 2)])
    wt_all = np.concatenate([fold_arrays[f]["wt"] for f in (0, 1, 2)])
    del fold_arrays
    gc.collect()

    team_s1 = pipeline_v4.EnsembleBinaryClf(mode="ens3", seed=0, use_gpu=use_gpu).fit(
        Xt_all, yt_all, sample_weight=wt_all
    )
    del Xt_all, yt_all, wt_all
    gc.collect()

    with open(args.decode_cfg) as f:
        v4_cfg = json.load(f)
    pipe = pipeline_v4.PipelineV4(
        imputer=imputer,
        dead_s1=dead_s1,
        team_s1=team_s1,
        stacker=stacker,
        decode_cfg=v4_cfg,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pipe.save(args.out)
    print(f"[{time.time()-t0:.1f}s] Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

