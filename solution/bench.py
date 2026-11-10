"""External benchmark (v3): the only place parameters are tuned or variants are compared.

    python -m solution.bench --exp E006 --method baseline --set window=50
    python -m solution.bench --exp E007 --method hmm --tune 150 --save solution/configs/E007.json

Cases: 3 external games x 3 pictures of what the ball does during a stoppage (external.SCENARIOS).
Each case is scored against TWO label variants (with and without our uncontrolled-touch rule) and
the objective is their mean, so nothing is tuned to one reading of the possession definition.

HEADLINE = HELD-OUT. With --tune, the search is repeated on every pair of games and scored on the
third; the mean of those three numbers is the `heldout` column and the number we quote. The
in-sample score of the final parameters is logged next to it, never instead of it.
Scored with the organisers' own evaluation/possession_metrics.py.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess

import numpy as np

from . import external, model
from .common import HERE, ROOT

import possession_metrics as pm  # noqa: E402  (path set up in common)

LOG = os.path.join(HERE, "experiments.csv")
FIELDS = ["exp", "when", "commit", "method", "note", "heldout", "insample", "spellF1_merged",
          "spellF1_raw", "macroF1", "poss_ratio", "aliveDeadF1", "teamAcc",
          "obj_g1", "obj_g2", "obj_g3", "obj_path", "obj_hold", "obj_hidden", "params"]

_CASES = None


def matches(seed=0):
    return {(g, sc): external.degrade(external.load(g), sc, seed)
            for g in external.GAMES for sc in external.SCENARIOS}


def cases():
    """[(game, scenario, features, gt_merged, gt_raw)] built once per process."""
    global _CASES
    if _CASES is None:
        _CASES = [(g, sc, model.features(m), m.gt, m.meta["gt_raw"])
                  for (g, sc), m in matches().items()]
    return _CASES


def score_pred(g, sc, pred, gt, gt_raw):
    r = pm.evaluate(gt, pred)
    raw = pm.evaluate(gt_raw, pred)["spell_f1_mean"]
    return dict(game=g, scenario=sc, spell=r["spell_f1_mean"], spell_raw=raw,
                obj=0.5 * (r["spell_f1_mean"] + raw), macro=r["macro_f1"],
                ratio=r["n_pred_spells"] / max(1, r["n_gt_spells"]),
                ad=r["alive_dead_f1"], team=r["team_acc_both_alive"])


def predict(method, F, params):
    if method == "baseline":
        return model.predict_baseline(F, int(params.get("window", 50)))
    if method == "hmm":
        return model.predict_hmm(F, params)
    raise ValueError(method)


def run(method, params, games=None):
    return [score_pred(g, sc, predict(method, F, params), gt, gt_raw)
            for g, sc, F, gt, gt_raw in cases() if games is None or g in games]


def mean(rows, key, **where):
    sel = [r[key] for r in rows if all(r[k] == v for k, v in where.items())]
    return float(np.mean(sel)) if sel else float("nan")


def summarise(rows):
    out = dict(insample=mean(rows, "obj"), spellF1_merged=mean(rows, "spell"),
               spellF1_raw=mean(rows, "spell_raw"), macroF1=mean(rows, "macro"),
               poss_ratio=mean(rows, "ratio"), aliveDeadF1=mean(rows, "ad"),
               teamAcc=mean(rows, "team"))
    for i, g in enumerate(external.GAMES, 1):
        out["obj_g%d" % i] = mean(rows, "obj", game=g)
    for sc in external.SCENARIOS:
        out["obj_%s" % sc] = mean(rows, "obj", scenario=sc)
    return out


def commit_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def log(exp, method, params, summary, note="", heldout=None):
    new = not os.path.isfile(LOG)
    with open(LOG, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        row = dict(exp=exp, when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   commit=commit_id(), method=method, note=note,
                   heldout="" if heldout is None else "%.4f" % heldout,
                   params=json.dumps(params, sort_keys=True))
        row.update({k: "%.4f" % v for k, v in summary.items()})
        writer.writerow(row)


def show(tag, s, heldout=None):
    print("%-18s HELD-OUT %s | in-sample %.4f (merged %.4f, raw %.4f)  macroF1 %.4f  poss x%.2f  "
          "aliveDead %.3f  team %.3f | games %.3f/%.3f/%.3f | path/hold/hidden %.3f/%.3f/%.3f" % (
              tag, "  n/a " if heldout is None else "%.4f" % heldout, s["insample"],
              s["spellF1_merged"], s["spellF1_raw"], s["macroF1"], s["poss_ratio"],
              s["aliveDeadF1"], s["teamAcc"], s["obj_g1"], s["obj_g2"], s["obj_g3"],
              s["obj_path"], s["obj_hold"], s["obj_hidden"]), flush=True)


# Search ranges for the random search: (low, high, log-scale?)
SPACE = dict(
    k_team=(0.3, 3.0, True), s_margin=(0.5, 5.0, True), r_near=(1.0, 8.0, True),
    v_gate=(4.0, 30.0, True), b_dead=(-2.0, 0.5, False), b_slow=(0.0, 1.5, False),
    b_slow_long=(0.0, 1.5, False), v_ref=(1.0, 3.0, False), c_team=(3.0, 60.0, True),
    c_dead=(3.0, 80.0, True), c_live=(3.0, 80.0, True), min_spell=(0, 75, False),
)


def sample(rng, base, scale=1.0):
    out = {}
    for k, (lo, hi, logscale) in SPACE.items():
        a, b = (np.log(lo), np.log(hi)) if logscale else (lo, hi)
        if scale < 1:
            c = np.log(base[k]) if logscale else base[k]
            half = 0.5 * scale * (b - a)
            a, b = max(a, c - half), min(b, c + half)
        v = rng.uniform(a, b)
        v = float(np.exp(v)) if logscale else float(v)
        out[k] = int(round(v)) if k == "min_spell" else round(v, 4)
    return out


def tune(n_iter, seed=0, games=None, fixed=None, verbose=True):
    """Random search (first half global, second half local around the incumbent)."""
    rng = np.random.default_rng(seed)
    fixed = fixed or {}
    best = dict(model.DEFAULTS, **fixed)
    best_score = mean(run("hmm", best, games), "obj")
    for i in range(n_iter):
        cand = dict(sample(rng, best, 1.0 if i < n_iter // 2 else 0.3), **fixed)
        score = mean(run("hmm", cand, games), "obj")
        if score > best_score:
            best, best_score = cand, score
            if verbose:
                print("  iter %3d  objective %.4f" % (i, score), flush=True)
    return best, best_score


def heldout_tune(n_iter, seed=0, fixed=None):
    held = []
    for g in external.GAMES:
        others = [x for x in external.GAMES if x != g]
        p, _ = tune(n_iter, seed, games=others, fixed=fixed, verbose=False)
        held.append(mean(run("hmm", p, [g]), "obj"))
        print("  held-out %s: %.4f" % (g, held[-1]), flush=True)
    return float(np.mean(held))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", default="hmm", choices=("baseline", "hmm"))
    ap.add_argument("--set", nargs="*", default=[], help="k=v parameter overrides (kept fixed)")
    ap.add_argument("--params", help="JSON file of parameters")
    ap.add_argument("--tune", type=int, default=0, help="random-search iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--save", help="write the final parameters to this JSON file")
    args = ap.parse_args()

    params = json.load(open(args.params)) if args.params else {}
    fixed = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    params.update(fixed)

    heldout = None
    if args.tune:
        heldout = heldout_tune(args.tune, args.seed, fixed)
        params, _ = tune(args.tune, args.seed, fixed=fixed)
    elif args.method == "baseline":
        # No fitted parameters, so the score on all games is already an out-of-sample score.
        heldout = mean(run(args.method, params), "obj")

    summary = summarise(run(args.method, params))
    show(args.exp, summary, heldout)
    log(args.exp, args.method, params, summary, args.note.strip(), heldout)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        json.dump(params, open(args.save, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
