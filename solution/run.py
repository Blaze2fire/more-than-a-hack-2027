"""Produce the submission CSV for the competition match.

    python -m solution.run --params solution/configs/final.json --name <team_name>

    python -m solution.run --learned --name <team_name>          # Option B (configs/gbm.pkl)

Writes evaluation/results/<name>.csv, validates it with the official checks, and prints an
aggregate summary of the file itself. It does NOT score against the public ground truth: an
earlier version printed that score on every run, which put a selection signal in front of us each
time we compared two variants (audit finding, RESEARCH_LOG E006). The alignment check lives in
solution/sanity.py and reports pass/fail only.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from . import atd, datasets, learned, model, pipeline_v4, tagging
from .common import D, LETTERS, ROOT, X

import possession_metrics as pm  # noqa: E402
import baseline as starter_baseline  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")
PIPELINE_V4_PATH = os.path.join(os.path.dirname(__file__), "configs", "pipeline_v4.pkl")
VITERBI4_CFG_PATH = os.path.join(os.path.dirname(__file__), "configs", "viterbi4.json")


def predict(m, params, method="learned", use_tags=False):
    if method == "learned":
        if os.path.exists(PIPELINE_V4_PATH):
            m_std = datasets.standardize_match(m)
            pipe = pipeline_v4.PipelineV4.load(PIPELINE_V4_PATH)
            if os.path.exists(VITERBI4_CFG_PATH):
                with open(VITERBI4_CFG_PATH) as f:
                    pipe.decode_cfg = json.load(f)
            codes = pipe.predict(m_std)
        else:
            cfg = json.load(open(os.path.join(os.path.dirname(learned.MODEL_PATH), "gbm_decode.json")))
            codes = learned.decode(learned.log_proba(learned.load(), m), m.blocks, cfg)
        codes[m.blocks[1][0]:atd.KICKOFF_2ND] = D
        return codes
    F = model.features(m)
    if method == "baseline":
        codes = model.predict_baseline(F, int(params.get("window", 50)))
    else:
        extra = tagging.priors(m, params) if use_tags and m.tags else None
        codes = model.predict_hmm(F, params, extra)
    # Published fact (FAQ.md): the 2nd half is graded from 110808 but kicks off at 113742.
    b0 = m.blocks[1][0]
    codes[b0:atd.KICKOFF_2ND] = D
    return codes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", help="JSON parameter file")
    ap.add_argument("--method", default="learned", choices=("hmm", "baseline", "learned"))
    ap.add_argument("--learned", action="store_true", help="shorthand for --method learned")
    ap.add_argument("--tags", action="store_true",
                    help="enable Smart Tagging priors (off by default: domain-shift safety)")
    ap.add_argument("--no-tags", action="store_false", dest="tags",
                    help="disable Smart Tagging priors (default)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    params = json.load(open(args.params)) if args.params else {}
    m = atd.load()
    codes = predict(m, params, "learned" if args.learned else args.method, args.tags)
    path = starter_baseline.write_submission(os.path.join(RESULTS, args.name + ".csv"),
                                             LETTERS[codes])

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, m.n)
    print("validation: %s" % (reason or "OK"))
    graded = np.zeros(m.n, dtype=bool)
    for b0, b1 in m.blocks:
        graded[b0:b1] = True
    runs = [(a, b) for a, b in m.blocks]
    stats = pm.possession_stats(codes, runs)
    share = {LETTERS[k]: round(100 * float(np.mean(codes[graded] == k)), 1) for k in (0, 1, 2, 3)}
    print("graded label share %%: %s" % share)
    print("possessions %d  mean %.1f s  home share %.2f" % (
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("sha256 %s" % hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])


if __name__ == "__main__":
    main()
