"""Alignment check against the public 5 minutes. Prints PASS / FAIL only, never a score.

    python -m solution.sanity evaluation/results/<name>.csv

The public ground truth may be used to check that a pipeline is correct (frames aligned, labels
the right way round) and for nothing else. So this script answers exactly those two questions and
shows no number that could be used to prefer one variant over another:

  orientation  the file agrees with the truth more often than the same file with H and A swapped
  alignment    the file agrees with the truth more often than the same file shifted by +/- 2 s
"""
import os
import sys

import numpy as np

from .common import A, H, ROOT, X

import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SHIFT = 50


def main():
    pred = pm.label_codes(pm.read_label_csv(sys.argv[1]))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = gt != X

    def agree(p):
        return float(np.mean(p[graded] == gt[graded]))

    base = agree(pred)
    swapped = agree(np.where(pred == H, A, np.where(pred == A, H, pred)))
    shifted = max(agree(np.roll(pred, SHIFT)), agree(np.roll(pred, -SHIFT)))
    print("orientation (H/A the right way round): %s" % ("PASS" if base > swapped else "FAIL"))
    print("alignment   (better than +/-%d frames): %s" % (SHIFT, "PASS" if base > shifted else "FAIL"))


if __name__ == "__main__":
    main()
