"""Score decoder / line-snap variants offline on predictions saved by
`deadline_benchmark evaluate --save-predictions` (held-out external fixtures only).

    .venv/bin/python -m solution.snap_eval --pred-dir solution/out/snap_base_eval --out solution/out/snap_results.json

Every case is regenerated exactly as the benchmark builds it (same degradation, same seed), so the
ball track used by the snap is the one the model saw. Prespecified primary comparison:
`restart_order` vs `restart_order+snap`; the other decoder rows are a disclosed secondary grid.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from solution import ablate_v4, bench_v4, datasets, deadline_benchmark as dbm, line_snap, viterbi4


def variants(base):
    fixed = dbm.decoder_variants(base)["restart_order"]
    out = {"restart_order": fixed}
    for s in (0.85, 0.7):
        out[f"cteam{s:g}"] = dict(fixed, c_team=fixed["c_team"] * s)
        out[f"cteam{s:g}_turn0"] = dict(fixed, c_team=fixed["c_team"] * s, min_turn=0, min_live_blip=10)
    return out


def case_match(raw, scenario, seed):
    if scenario.startswith("gap-stress"):
        return ablate_v4.make_elvern_burst_match(datasets.degrade_atd(raw, "natural", seed),
                                                 raw.name.replace("dfl_", ""), scenario)
    if scenario == "moving_dead":
        return dbm.coherent_carried_dead(raw, seed)
    return datasets.degrade_atd(raw, scenario, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--mix", default="nn0.7_ownership")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neural", help="re-blend saved GBDT predictions with this held-out-safe NN checkpoint")
    ap.add_argument("--neural-weight", type=float, default=0.7)
    args = ap.parse_args()
    models = None
    skip_metrica = False
    if args.neural:
        import pickle
        from solution import neural_v2
        from solution.ownership_blend import blend_logp
        man = json.load(open(args.neural.replace("models_", "manifest_").replace(".pkl", ".json")))
        if set(man["train_keys"]) & {"dfl4", "dfl5", "dfl6", "g"}:
            raise ValueError("Neural checkpoint saw a held-out fixture")
        skip_metrica = any(k.startswith("met") for k in man["train_keys"])  # trained on Metrica: score DFL + Groningen only
        models = pickle.load(open(args.neural, "rb"))
    base = dbm.frozen_baseline_config()
    decs = variants(base)
    rows = []
    for raw in dbm.external_cases([1, 2, 3]):
        provider = "DFL" if raw.name.startswith("dfl_") else ("Groningen" if raw.name == "groningen" else "Metrica")
        if provider == "Metrica" and skip_metrica:
            continue
        for f in sorted(os.listdir(args.pred_dir)):
            if not (f.startswith(raw.name + "__") and f.endswith(f"__{args.mix}.npz")):
                continue
            _, scenario, seed, _ = f[:-4].split("__")
            seed = int(seed[1:])
            m = case_match(raw, scenario, seed)
            z = np.load(os.path.join(args.pred_dir, f))
            lp, gt, gt_merged = z["lp"], z["gt"], z["gt_merged"]
            if models is not None:
                zg = np.load(os.path.join(args.pred_dir, f.replace(f"__{args.mix}.npz", "__gbdt.npz")))
                p_nn, _ = neural_v2.predict_match(models, m)
                lp = blend_logp(zg["lp"], p_nn, args.neural_weight)
                assert np.array_equal(lp[:, 2], zg["lp"][:, 2])
            assert np.array_equal(gt, m.gt), "regenerated case does not match saved predictions"
            for dname, cfg in decs.items():
                pred = viterbi4.decode4(lp, m.blocks, cfg)
                for snap in (False, True):
                    p = line_snap.snap_onsets(pred, m.ball, m.pitch, m.blocks) if snap else pred
                    row = bench_v4.score_match(m.name, scenario, p, gt, gt_merged)
                    row.update(ablate_v4.compute_short_and_trans_metrics(p, gt, m.blocks))
                    row.update(provider=provider, seed=seed, variant=dname + ("+snap" if snap else ""),
                               n_changed=int((p != pred).sum()))
                    rows.append(row)
            print("scored", f, flush=True)
    df = pd.DataFrame(rows)
    cols = ["spell_f1", "spell_f1_merged", "alive_dead_f1", "short_2s_rec", "trans_f1_12", "n_pred_spells", "n_changed"]
    cell = df.groupby(["variant", "provider", "scenario"])[cols].mean()
    prov = cell.groupby(["variant", "provider"]).mean()
    pb = prov.groupby("variant").mean()
    pb_dg = prov[prov.index.get_level_values("provider").isin(["DFL", "Groningen"])].groupby("variant").mean()
    print("\nProvider-balanced (all providers):\n", pb.round(4).to_string())
    print("\nProvider-balanced (DFL + Groningen; Metrica dead labels derive from ball absence):\n",
          pb_dg[["spell_f1", "alive_dead_f1", "trans_f1_12"]].round(4).to_string())
    print("\nPer provider spell_f1:\n", prov["spell_f1"].unstack().round(4).to_string())
    print("\nPer provider x scenario spell_f1 (restart_order vs +snap):\n",
          cell["spell_f1"].unstack(0)[["restart_order", "restart_order+snap"]].round(4).to_string())
    with open(args.out, "w") as fh:
        json.dump(dict(protocol="held-out external; saved benchmark predictions; prespecified snap DEFAULTS",
                       snap_defaults=line_snap.DEFAULTS, decoders=decs, rows=rows), fh, default=float)


if __name__ == "__main__":
    main()
