"""Meaningful guardrails for learned boundaries and frozen-mask ownership decoding."""
import itertools
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from solution import changepoint, pipeline_v4, viterbi4
from solution.common import A, D, H, X


class BoundaryTests(unittest.TestCase):
    def test_targets_ignore_unknown_and_cross_block_changes(self):
        gt = np.array([H, H, A, A, X, H, A, A])
        target, valid = changepoint.boundary_targets(gt, [(0, 5), (5, 8)], radius=0)
        np.testing.assert_array_equal(np.flatnonzero(target), [2, 6])
        self.assertFalse(valid[4])
        gt = np.array([H, H, A, A])
        target, _ = changepoint.boundary_targets(gt, [(0, 2), (2, 4)], radius=0)
        self.assertEqual(int(target.sum()), 0)

    def test_features_exact_swap_and_block_isolation(self):
        rng = np.random.default_rng(7)
        xh, xa = rng.normal(size=(2, 80, 156)).astype(np.float32)
        blocks = [(0, 40), (40, 80)]
        features = changepoint.boundary_features(xh, xa, blocks)
        np.testing.assert_array_equal(features, changepoint.boundary_features(xa, xh, blocks))
        xh[40:] += 100
        xa[40:] -= 100
        np.testing.assert_array_equal(features[:40], changepoint.boundary_features(xh, xa, blocks)[:40])

    def test_ownership_pass_optimal_and_preserves_dead_unknown(self):
        rng = np.random.default_rng(11)
        for _ in range(15):
            lp = rng.normal(size=(8, 3))
            codes = np.array([H, H, H, H, D, X, A, A], dtype=np.int8)
            out = viterbi4.ownership_pass(lp, codes, [(0, 8)], c_team=0.7)
            np.testing.assert_array_equal(out[4:6], codes[4:6])
            paths = list(itertools.product((H, A), repeat=4))
            def score(path):
                return sum(lp[t, state] for t, state in enumerate(path)) - 0.7 * sum(a != b for a, b in zip(path, path[1:]))
            self.assertAlmostEqual(score(out[:4]), max(map(score, paths)))
            swapped = viterbi4.ownership_pass(lp[:, [A, H, D]], np.where(codes == H, A, np.where(codes == A, H, codes)), [(0, 8)], 0.7)
            np.testing.assert_array_equal(swapped, np.where(out == H, A, np.where(out == A, H, out)))

    def test_packaged_frozen_decoder_equals_benchmark_and_retains_reference(self):
        n = 80
        m = SimpleNamespace(blocks=[(0, 80)])
        pd = np.r_[np.full(30, .05), np.full(20, .99), np.full(30, .05)].astype(np.float32)
        ph = np.linspace(.15, .85, n, dtype=np.float32)
        neural = np.r_[np.full(40, .95), np.full(40, .05)].astype(np.float32)
        cfg = dict(viterbi4.DECODE4_DEFAULTS, min_dead=0, min_live_blip=0, min_turn=0)
        pipe = pipeline_v4.PipelineV4(None, None, None, decode_cfg=dict(cfg, freeze_dead_from_gbdt=True),
                                      neural_weight=.7, dead_reference_cfg=cfg)
        inputs = (None, None, pd, ph, None, neural, None,
                  np.zeros((n, 40)), np.zeros((n, 65)), np.zeros((n, 65)))
        with patch.object(pipe, "_compute_s1", return_value=inputs):
            actual = pipe.predict(m)
            mixed, _ = pipe._combine_stacker_and_neural(m, pd, ph, neural, None, *inputs[-3:])
            base, _ = pipe._combine_stacker_and_neural(m, pd, ph, None, None, *inputs[-3:])
        reference = viterbi4.decode4(base, m.blocks, cfg)
        expected = viterbi4.ownership_pass(mixed, reference, m.blocks, cfg["c_team"] * .5)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual == D, reference == D)
        with self.assertRaises(ValueError):
            viterbi4.decode4(mixed, m.blocks, pipe.decode_cfg)


if __name__ == "__main__":
    unittest.main()
