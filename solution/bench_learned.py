"""Benchmark Option B (learned emissions + Viterbi) on benchmark v3, leave-one-game-out.

    python -m solution.bench_learned --exp E008 [--tune 200] [--final]

HEADLINE = HELD-OUT at both levels: for each held-out game the classifiers are trained on the
other two games, AND the decoder costs are tuned on the other two games' out-of-fold
probabilities. The in-sample number (decoder tuned on all three) is logged next to it.
--final additionally trains on all three games and saves the models + decoder config.
"""
import argparse
import json
import os

import numpy as np

from . import bench, external, learned
from .common import CACHE_DIR, HERE

SPACE = dict(w=(0.3, 3.0), c_team=(1.0, 60.0), c_dead=(1.0, 80.0), c_live=(1.0, 80.0))


def oof(matches, tag, **fit_kw):
    """Out-of-fold log-probabilities: each game predicted by models trained on the other two."""
    path = os.path.join(CACHE_DIR, "oof_%s.npz" % tag)
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z["%s|%s" % k] for k in matches}
    out = {}
    for g in external.GAMES:
        clfs = learned.fit([m for (gg, _), m in matches.items() if gg != g], **fit_kw)
        for (gg, sc), m in matches.items():
            if gg == g:
                out[(gg, sc)] = learned.log_proba(clfs, m).astype(np.float32)
        print("  fold %s trained" % g, flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **{"%s|%s" % k: v for k, v in out.items()})
    return out


def score(matches, logp, cfg, games=None):
    return [bench.score_pred(g, sc, learned.decode(logp[(g, sc)], m.blocks, cfg), m.gt,
                             m.meta["gt_raw"])
            for (g, sc), m in matches.items() if games is None or g in games]


def tune(matches, logp, n_iter, seed=0, games=None, verbose=True):
    rng = np.random.default_rng(seed)
    best = dict(learned.DECODE_DEFAULTS)
    best_s = bench.mean(score(matches, logp, best, games), "obj")
    for i in range(n_iter):
        cand = {}
        for k, (lo, hi) in SPACE.items():
            a, b = np.log(lo), np.log(hi)
            if i >= n_iter // 2:
                c, half = np.log(best[k]), 0.15 * (b - a)
                a, b = max(a, c - half), min(b, c + half)
            cand[k] = round(float(np.exp(rng.uniform(a, b))), 4)
        cand["min_spell"] = int(rng.integers(0, 60))
        s = bench.mean(score(matches, logp, cand, games), "obj")
        if s > best_s:
            best, best_s = cand, s
            if verbose:
                print("  iter %3d  objective %.4f" % (i, s), flush=True)
    return best, best_s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tune", type=int, default=200)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    matches = bench.matches()
    logp = oof(matches, args.exp)

    held = []
    for g in external.GAMES:
        others = [x for x in external.GAMES if x != g]
        c, _ = tune(matches, logp, args.tune, games=others, verbose=False)
        held.append(bench.mean(score(matches, logp, c, [g]), "obj"))
        print("  held-out %s: %.4f" % (g, held[-1]), flush=True)
    heldout = float(np.mean(held))

    cfg, _ = tune(matches, logp, args.tune)
    summary = bench.summarise(score(matches, logp, cfg))
    bench.show(args.exp, summary, heldout)
    bench.log(args.exp, "gbm2+viterbi", cfg, summary, args.note.strip(), heldout)

    if args.final:
        learned.save(learned.fit(list(matches.values())))
        json.dump(cfg, open(os.path.join(HERE, "configs", "gbm_decode.json"), "w"), indent=1,
                  sort_keys=True)
        print("saved final models and decoder config")


if __name__ == "__main__":
    main()
