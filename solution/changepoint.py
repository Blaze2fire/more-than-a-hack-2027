"""External-label H/A boundary head; invariant to the names of the two teams.

Labels are used only by ``fit``. Inference reads tracking features only and the
result discounts live H/A transitions, never the dead-ball classifier.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from solution.common import A, H, X


def boundary_features(x_home, x_away, blocks):
    """Symmetric spatial/contact evidence and local motion, reset at half time."""
    # The original 156 columns remain compatible with the frozen baseline.
    xh = np.nan_to_num(x_home[:, :156].astype(np.float32), posinf=100, neginf=-100)
    xa = np.nan_to_num(x_away[:, :156].astype(np.float32), posinf=100, neginf=-100)
    symmetric = np.column_stack((0.5 * (xh + xa), np.abs(xh - xa)))
    # Compact temporal contrast uses the 65 base channels, with no cross-block lookups.
    contrasts = []
    for lag in (5, 15, 30):
        contrast = np.zeros((len(xh), 65), dtype=np.float32)
        for start, stop in blocks:
            idx = np.arange(start, stop)
            left = np.maximum(start, idx - lag)
            right = np.minimum(stop - 1, idx + lag)
            dh = xh[right, :65] - xh[left, :65]
            da = xa[right, :65] - xa[left, :65]
            contrast[start:stop] = 0.5 * (np.abs(dh) + np.abs(da))
        contrasts.append(contrast)
    return np.clip(np.column_stack((symmetric, *contrasts)), -1e4, 1e4)


def boundary_targets(gt, blocks, radius=10):
    """Native live H/A transitions only; unknown labels are never negatives."""
    target = np.zeros(len(gt), dtype=np.int8)
    valid = np.zeros(len(gt), dtype=bool)
    for start, stop in blocks:
        g = gt[start:stop]
        live = (g == H) | (g == A)
        change = np.zeros(len(g), dtype=np.int8)
        change[1:] = live[1:] & live[:-1] & (g[1:] != g[:-1])
        target[start:stop] = maximum_filter1d(change, size=2 * radius + 1, mode="constant")
        # Do not teach unknown/provider-unlabeled frames as no-change evidence.
        valid[start:stop] = g != X
    return target, valid


@dataclass
class BoundaryModel:
    classifier: object
    training_fixtures: tuple

    def predict_from_features(self, x_home, x_away, blocks):
        features = boundary_features(x_home, x_away, blocks)
        return self.classifier.predict_proba(features)[:, 1].astype(np.float32)


def fit(records, seed=0, stride=5, threads=4):
    """Fit on explicit (fixture_name, Xh, Xa, gt, blocks) training records only."""
    from lightgbm import LGBMClassifier

    xs, ys = [], []
    names = []
    for name, xh, xa, gt, blocks in records:
        target, valid = boundary_targets(gt, blocks)
        sampled = np.zeros(len(gt), dtype=bool)
        for start, stop in blocks:
            sampled[start:stop:stride] = True
        keep = valid & sampled
        xs.append(boundary_features(xh, xa, blocks)[keep])
        ys.append(target[keep])
        names.append(name)
    if not xs or len(np.unique(np.concatenate(ys))) != 2:
        raise ValueError("Boundary fitting requires external training data with live team changes")
    model = LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        min_child_samples=80, reg_lambda=3.0, colsample_bytree=0.8,
        n_jobs=threads, random_state=seed, verbose=-1,
    )
    model.fit(np.vstack(xs), np.concatenate(ys))
    return BoundaryModel(model, tuple(sorted(set(names))))
