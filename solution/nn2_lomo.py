"""Held-out training/prediction harness for neural_v2 (external data only).

Each ';'-separated group in --groups is trained once with every member held out, then predicts
the same degraded evaluation matches that solution.ablate_v4 scores (identical seeds), so the
outputs line up frame-for-frame with the GBDT out-of-fold files for blending (solution.blend_eval).

    .venv/bin/python -m solution.nn2_lomo --tag dev --groups "4,5,6,g"        # one model, quick
    .venv/bin/python -m solution.nn2_lomo --tag lomo --groups "0;1;2;3;4;5;6;g"  # full LOMO
    .venv/bin/python -m solution.nn2_lomo --tag final --groups "" --save-final solution/configs/neural_v2.pkl
"""
import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np

from solution import ablate_v4, datasets, external, neural_v2
from solution.common import OUT_DIR

def source_fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("neural_v2.py", "neural_team.py", "ball_tracker.py", "datasets.py"):
        with open(os.path.join(os.path.dirname(__file__), name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


SOURCE_FINGERPRINT = source_fingerprint()
CACHE = os.path.join(OUT_DIR, "nn2_cache_" + SOURCE_FINGERPRINT[:12])


def _train_record(key: str, loader, weight: float, seed: int) -> dict:
    path = os.path.join(CACHE, f"train_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            rec = pickle.load(f)
    else:
        m = loader()
        c = neural_v2.compact_match(m)
        alt = neural_v2.compact_match(datasets.degrade_atd(m, scenario="atd_blind", seed=seed))
        rec = neural_v2.prepare_record(c, alt=dict(P=alt["P"], M=alt["M"]), meta=m.meta)
        with open(path, "wb") as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    rec["weight"] = weight
    rec["key"] = key
    return rec


def load_train_records(use_metrica: bool, n_sc: int, w_met: float, w_sc: float,
                       train_keys=None) -> list:
    os.makedirs(CACHE, exist_ok=True)
    recs = []
    for k, mid in enumerate(datasets.DFL_GAMES):
        if train_keys is None or f"dfl{k}" in train_keys:
            recs.append(_train_record(f"dfl{k}", lambda mid=mid: datasets.load_dfl(mid), 1.0, 7000 + k))
    if train_keys is None or "g" in train_keys:
        recs.append(_train_record("g", datasets.load_groningen, 1.0, 7100))
    if use_metrica:
        for i, g in enumerate(external.GAMES):
            if train_keys is None or f"met{i}" in train_keys:
                recs.append(_train_record(f"met{i}", lambda g=g: datasets.standardize_match(external.load(g)), w_met, 7200 + i))
    for i, mid in enumerate(datasets.list_skillcorner_current()[:n_sc]):
        if train_keys is None or f"sc{i}" in train_keys:
            recs.append(_train_record(f"sc{i}", lambda mid=mid: datasets.load_skillcorner_current(mid), w_sc, 7300 + i))
    if train_keys is not None and {r["key"] for r in recs} != set(train_keys):
        raise ValueError("Unknown or unavailable training keys (check --no-metrica / --n-sc)")
    return recs


def eval_compacts(member: str):
    """Yield (file_stem, compact dict) for one held-out member, matching ablate_v4's eval matches."""
    if member == "g":
        gron = datasets.load_groningen()
        for sc in ("natural", "atd_real"):
            yield f"groningen__{sc}", neural_v2.compact_match(datasets.degrade_atd(gron, scenario=sc, seed=199))
        return
    k = int(member)
    m_raw = datasets.load_dfl(datasets.DFL_GAMES[k])
    for sc in ("natural", "atd_real", "atd_blind", "atd_carried_dead"):
        yield f"{m_raw.name}__{sc}", neural_v2.compact_match(datasets.degrade_atd(m_raw, scenario=sc, seed=100 + k))
    if m_raw.name in ("dfl_J03WPY", "dfl_J03WQQ", "dfl_J03WR9"):
        m_nat = datasets.degrade_atd(m_raw, scenario="natural", seed=100 + k)
        m_b = ablate_v4.make_elvern_burst_match(m_nat, fixture=m_raw.name.replace("dfl_", ""))
        yield f"{m_raw.name}__gap-stress-burst-01", neural_v2.compact_match(m_b)


def cached_eval_compacts(member: str):
    path = os.path.join(CACHE, f"eval_{member}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = [(stem, dict(P=c["P"], M=c["M"], ball=c["ball"], blocks=c["blocks"])) for stem, c in eval_compacts(member)]
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--groups", default="4,5,6,g")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seeds", default="0")
    p.add_argument("--no-metrica", action="store_true")
    p.add_argument("--w-met", type=float, default=0.7)
    p.add_argument("--n-sc", type=int, default=0)
    p.add_argument("--w-sc", type=float, default=0.15)
    p.add_argument("--save-final", default="")
    p.add_argument("--train-keys", default="", help="Explicit source whitelist, e.g. dfl0,dfl1,dfl2,dfl3")
    p.add_argument("--frame-stride", type=int, default=1, help="Sample interval at training/inference; 5 = 5 Hz")
    p.add_argument("--gap-policy", choices=("mixed", "short", "none"), default="mixed")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    t0 = time.time()
    neural_v2.torch.set_num_threads(args.threads)
    train_keys = set(args.train_keys.split(",")) if args.train_keys else None
    recs = load_train_records(not args.no_metrica, args.n_sc, args.w_met, args.w_sc, train_keys=train_keys)
    print(f"[{time.time() - t0:.0f}s] {len(recs)} training records loaded", flush=True)
    out_dir = args.out_dir or os.path.join(OUT_DIR, f"nn2_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    for group in args.groups.split(";"):
        members = [g for g in group.split(",") if g]
        held = {f"dfl{g}" if g != "g" else "g" for g in members}
        tr = [r for r in recs if r["key"] not in held]
        if not tr:
            raise ValueError("No records remain after holding out this group")
        if {r["key"] for r in tr} & held:
            raise AssertionError("A held-out fixture entered neural training")
        models = []
        for sd in seeds:
            models.append(neural_v2.fit(tr, steps=args.steps, batch=args.batch, seed=sd, width=args.width,
                                        gru_layers=args.gru_layers, lr=args.lr, log_every=max(1, min(100, args.steps)),
                                        frame_stride=args.frame_stride, gap_policy=args.gap_policy))
            print(f"[{time.time() - t0:.0f}s] group [{group}] seed {sd} trained on {len(tr)} records", flush=True)
        group_key = group.replace(",", "_") or "all"
        model_path = os.path.join(out_dir, f"models_{group_key}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = dict(
            tag=args.tag, source_fingerprint=SOURCE_FINGERPRINT, args=vars(args),
            train_keys=sorted(r["key"] for r in tr), held_keys=sorted(held),
            training_fixtures=[r["name"] for r in tr],
            fit_scope="public external tracking and native provider labels only; no learned imputer",
            sampling_fps=25 / args.frame_stride,
            model_sha256=hashlib.sha256(open(model_path, "rb").read()).hexdigest(),
        )
        with open(os.path.join(out_dir, f"manifest_{group_key}.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if args.save_final and not members:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_final)), exist_ok=True)
            with open(args.save_final, "wb") as f:
                pickle.dump(models, f)
            print("saved", args.save_final, flush=True)
        for g in members:
            for stem, c in cached_eval_compacts(g):
                p_ha, p_sw = neural_v2.predict_compact(models, c)
                np.savez(os.path.join(out_dir, stem + ".npz"), p_ha_nn=p_ha, p_sw_nn=p_sw)
            print(f"[{time.time() - t0:.0f}s] predicted held-out {g}", flush=True)


if __name__ == "__main__":
    main()
