"""Where is the score lost? Oracle ablations on the external benchmark.

    python -m solution.diagnose [params.json]

  oracle-dead   true dead mask, our team decoding   -> what perfect dead-ball detection would give
  oracle-team   our dead mask, true team             -> what perfect team choice would give
"""
import json
import sys

import numpy as np

from . import bench, model
from .common import D, X

import possession_metrics as pm  # noqa: E402


def main():
    params = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    rows = {"ours": [], "oracle-dead": [], "oracle-team": []}
    for g, sc, F, gt in bench.cases():
        pred = model.predict_hmm(F, params)
        graded = gt != X
        # oracle dead: decode team only (dead impossible), then stamp the true dead mask
        cfg = dict(model.DEFAULTS, **params)
        llr, _ = model.evidence(F, cfg)
        team_only = np.full(F["n"], X, dtype=np.int8)
        for b0, b1 in F["blocks"]:
            l = llr[b0:b1]
            path = model.viterbi3(0.5 * l, -0.5 * l, np.full(len(l), -1e6), cfg["c_team"], 1e9, 1e9)
            team_only[b0:b1] = path
        od = np.where(gt == D, D, team_only)
        od[~graded] = X
        # oracle team: where we say alive and truth is alive, use the true team
        ot = pred.copy()
        both = (pred != D) & (gt != D) & graded
        ot[both] = gt[both]
        for key, p in (("ours", pred), ("oracle-dead", od), ("oracle-team", ot)):
            r = pm.evaluate(gt, p)
            rows[key].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"] / r["n_gt_spells"]))
    for key, v in rows.items():
        a = np.mean(v, axis=0)
        print("%-12s spellF1 %.4f  macroF1 %.4f  poss x%.2f" % (key, a[0], a[1], a[2]))


if __name__ == "__main__":
    main()
