"""Blend held-out GBDT out-of-fold log-probs with held-out neural ownership and score the decode.

    .venv/bin/python -m solution.blend_eval --oof-dir solution/out/oof_G0 --nn-dir solution/out/nn2_dev \
        --weights 0,0.4,0.6,0.8,1.0

Only (match, scenario) files present in --nn-dir are scored, so a quick dev group and a full LOMO
use the same command. P(dead) is never touched by the blend. External matches only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os

import numpy as np
import pandas as pd

from solution import ablate_v4, bench_v4, viterbi4
from solution.common import A, D, H, ROOT
from solution.ownership_blend import blend_logp


def load_case(oof_dir: str, nn_dir: str, stem: str, w_nn: float, use_sw: bool = True) -> dict:
    z = np.load(os.path.join(oof_dir, stem + ".npz"), allow_pickle=False)
    lp, sw = z["lp"], (z["sw_sig"] if z["sw_sig"].size else None)
    if nn_dir:
        q = np.load(os.path.join(nn_dir, stem + ".npz"))
        if w_nn > 0:
            lp = blend_logp(lp, q["p_ha_nn"], w_nn)
        if use_sw:
            sw = q["p_sw_nn"]
    return dict(lp=lp, sw=sw, gt=z["gt"].astype(np.int8), gt_merged=z["gt_merged"].astype(np.int8),
                blocks=[tuple(b) for b in z["blocks"].tolist()], gap=z["rb_gap"])


def _score(task):
    oof_dir, nn_dir, stem, w_nn, cfg, use_sw = task
    c = load_case(oof_dir, nn_dir, stem, w_nn, use_sw)
    game, sc = stem.split("__", 1)
    pred = viterbi4.decode4(c["lp"], c["blocks"], cfg, sw_signal=c["sw"])
    row = bench_v4.score_match(game, sc, pred, c["gt"], gt_merged=c["gt_merged"])
    row.update(ablate_v4.compute_short_and_trans_metrics(pred, c["gt"], c["blocks"]))
    gt = c["gt"]
    for rk, msk in (("vis", c["gap"] == 0), ("burst", c["gap"] > 25)):
        lb = msk & (pred != D) & ((gt == H) | (gt == A))
        row[f"{rk}_team_acc"] = float(np.mean(pred[lb] == gt[lb])) if lb.any() else np.nan
    row["oracle_dead"] = row["oracle_team"] = np.nan
    return row


def run(oof_dir, nn_dir, weights, cfg, use_sw=True, workers=2):
    src = nn_dir or oof_dir
    stems = sorted(f[:-4] for f in os.listdir(src) if f.endswith(".npz"))
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for w in weights:
            rows = list(ex.map(_score, [(oof_dir, nn_dir, s, w, cfg, use_sw) for s in stems]))
            out[w] = (ablate_v4.aggregate_summary(pd.DataFrame(rows)), rows)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-dir", required=True)
    p.add_argument("--nn-dir", default="")
    p.add_argument("--weights", default="0,0.4,0.6,0.8,1.0")
    p.add_argument("--decode-cfg", default=os.path.join(ROOT, "solution/configs/viterbi4.json"))
    p.add_argument("--no-sw", action="store_true")
    p.add_argument("--out-json", default="")
    p.add_argument("--save-blended-dir", default="", help="write blended lp/sw files (first weight) for solution.tune_v4")
    args = p.parse_args()
    if args.save_blended_dir:
        os.makedirs(args.save_blended_dir, exist_ok=True)
        w0 = float(args.weights.split(",")[0])
        for f in sorted(os.listdir(args.nn_dir)):
            if f.endswith(".npz"):
                c = load_case(args.oof_dir, args.nn_dir, f[:-4], w0, not args.no_sw)
                np.savez(os.path.join(args.save_blended_dir, f), lp=c["lp"], gt=c["gt"], gt_merged=c["gt_merged"],
                         sw_sig=c["sw"] if c["sw"] is not None else np.array([], dtype=np.float32),
                         blocks=np.array(c["blocks"], dtype=np.int32), rb_gap=c["gap"])
        print("saved blended OOF (w_nn=%.2f) to %s" % (w0, args.save_blended_dir))
        return
    with open(args.decode_cfg) as f:
        cfg = json.load(f)
    res = run(args.oof_dir, args.nn_dir, [float(w) for w in args.weights.split(",")], cfg, use_sw=not args.no_sw)
    keys = None
    for w, (summ, _) in res.items():
        keys = keys or list(summ.keys())
        print(f"w_nn={w:.2f} | " + " | ".join(f"{k.replace('dfl_', '')}={summ[k]['spell_f1']:.4f}" for k in keys), flush=True)
    for w, (summ, _) in res.items():
        print(f"w_nn={w:.2f} team_acc | " + " | ".join(f"{k.replace('dfl_', '')}={summ[k]['team_acc']:.4f}" for k in keys))
        print(f"w_nn={w:.2f} short2s  | " + " | ".join(f"{k.replace('dfl_', '')}={summ[k]['short_2s_rec']:.3f}" for k in keys))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({str(w): {"summary": s, "rows": r} for w, (s, r) in res.items()}, f, indent=2)


if __name__ == "__main__":
    main()
