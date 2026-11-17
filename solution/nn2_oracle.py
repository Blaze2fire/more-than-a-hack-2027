"""Ownership capacity diagnostic using the TRUE external dead mask (never production inference)."""
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from solution import ablate_v4, datasets, model
from solution.common import A, D, H, X


def oracle_dead_decode(ph, gt, blocks, switch_cost=3.6):
    """Fix D from native GT, then independently decode every contiguous true-live run."""
    out = np.full(len(gt), X, dtype=np.int8)
    for b0, b1 in blocks:
        out[b0:b1] = D
        live = (gt[b0:b1] == H) | (gt[b0:b1] == A)
        edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for a, b in zip(edges[::2], edges[1::2]):
            a, b = b0 + int(a), b0 + int(b)
            p = np.clip(ph[a:b].astype(np.float64), 1e-6, 1 - 1e-6)
            labels = model.viterbi3(np.log(p), np.log1p(-p), np.full(len(p), -1e9), switch_cost, 0, 0)
            if (labels == D).any():
                raise AssertionError("Oracle live run unexpectedly decoded as dead")
            out[a:b] = labels
        out[b0:b1][gt[b0:b1] == X] = X
    return out


def main():
    rows = []
    for policy, directory in (("mixed", "deadline_neural"), ("short", "deadline_neural_short")):
        directory = Path("solution/out") / directory
        manifest = json.loads((directory / "manifest_4_5_6_g.json").read_text())
        if manifest["train_keys"] != ["dfl0", "dfl1", "dfl2", "dfl3"]:
            raise ValueError("Oracle diagnostic requires first-four-only neural fitting")
        for fixture in ("J03WPY", "J03WQQ", "J03WR9", "groningen"):
            m = datasets.load_groningen() if fixture == "groningen" else datasets.load_dfl(fixture)
            scenarios = ("natural",) if fixture == "groningen" else ("natural", "gap-stress-burst-01")
            for scenario in scenarios:
                path = directory / (m.name + "__" + scenario + ".npz")
                with np.load(path, allow_pickle=False) as z:
                    ph = z["p_ha_nn"]
                if len(ph) != m.n:
                    raise ValueError("External prediction/label length mismatch")
                pred = oracle_dead_decode(ph, m.gt, m.blocks)
                scores = pm.evaluate(m.gt, pred, bootstrap_n=0)
                row = dict(policy=policy, fixture=fixture, scenario=scenario, model_sha256=manifest["model_sha256"],
                           spell_f1=scores["spell_f1_mean"], macro_f1=scores["macro_f1"],
                           team_acc=scores["team_acc_both_alive"], n_pred_spells=scores["n_pred_spells"],
                           n_gt_spells=scores["n_gt_spells"])
                row.update(ablate_v4.compute_short_and_trans_metrics(pred, m.gt, m.blocks))
                rows.append(row)
    result = dict(
        protocol="ORACLE diagnostic only: true external D mask, reset at every true live-run boundary, fixed H/A cost3.6, no smoothing, no duration filtering, no change head",
        caveat="Not production performance, not directly comparable to participant full pipelines; no Gron burst cache exists",
        rows=rows,
    )
    Path("solution/out/deadline_neural_oracle.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
