"""Data-boundary and preprocessing invariants; uses synthetic matches only."""
import dataclasses
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from solution import ball_tracker, datasets, external, features_v4, pipeline_v4, train_v4
from solution.common import A, D, H, X, Match


def fixture(name="synthetic", n=400, source="sportec-dfl-v1", dead=True):
    t = np.arange(n, dtype=np.float32)
    pos = np.empty((n, 4, 2), dtype=np.float32)
    for j in range(4):
        pos[:, j, 0] = 20 + j * 15 + 0.02 * t
        pos[:, j, 1] = 15 + j * 8 + np.sin(t / 60)
    blocks = [(0, n // 2 - 10), (n // 2 + 10, n)]
    gt = np.full(n, X, dtype=np.int8)
    for a, b in blocks:
        gt[a:b] = np.where(np.arange(b - a) % 120 < 60, H, A)
        if dead:
            gt[a + 35:a + 55] = D
    ball = np.column_stack([42 + 0.03 * t, 33 + np.sin(t / 20)]).astype(np.float32)
    return Match(name=name, ball=ball, pos=pos, spd=datasets.compute_speed(pos, blocks),
                 team=np.array([H, H, A, A], dtype=np.int8), blocks=blocks, gt=gt,
                 meta=dict(source=source, has_real_dead_ball=dead, gt_raw=gt.copy()))


class TrainingHygieneTests(unittest.TestCase):
    def test_per_fixture_seeds_and_view_groups(self):
        matches = [fixture("one"), fixture("two"), fixture("three")]
        specs = train_v4.build_training_specs(matches, seed=9, long_bursts=True,
                                             stoppage_augmentation=True)
        self.assertEqual(len({s.seed for s in specs}), len(specs))
        for m in matches:
            own = [s for s in specs if s.fixture == m.name]
            self.assertEqual(len({s.fold for s in own}), 1)
        self.assertEqual(train_v4.fixture_seed("one", "path", 9),
                         train_v4.fixture_seed("one", "path", 9))
        with self.assertRaises(ValueError):
            train_v4.build_training_specs([matches[0], matches[0]])

    def test_entire_preprocessor_excludes_oof_fixture(self):
        matches = [fixture("one"), fixture("two"), fixture("three"),
                   fixture("sc", source="skillcorner-current", dead=False)]
        seen = []
        def fake_imputer(ms, **kwargs):
            return frozenset(m.name for m in ms)
        def fake_stage1(specs, imputer, *args, **kwargs):
            trained = frozenset(s.fixture for s in specs)
            self.assertEqual(trained, imputer)
            seen.append(trained)
            def predict(m):
                self.assertNotIn(m.name, trained)
                return (np.full(m.n, .2, np.float32), np.full(m.n, .6, np.float32),
                        np.zeros((m.n, 40), np.float32), np.zeros((m.n, 65), np.float32),
                        np.zeros((m.n, 65), np.float32))
            return SimpleNamespace(predict_stage1_for_stacker=predict)
        def fake_stack(items, **kwargs):
            names = [item[5].name for item in items]
            self.assertIn("sc", names)
            return SimpleNamespace()
        with patch.object(ball_tracker, "fit_constellation_imputer", fake_imputer), \
             patch.object(train_v4, "_fit_stage1", fake_stage1), \
             patch.object(pipeline_v4, "fit_temporal_stacker_from_oof", fake_stack):
            _, provenance = train_v4.fit_pipeline(matches, progress=lambda *a, **k: None)
        self.assertEqual(len(seen), 4)  # 3 OOF fits + separate all-data final fit
        for fold in provenance["inner_folds"]:
            self.assertTrue(set(fold["imputer_train"]).isdisjoint(fold["stage2_oof"]))
            self.assertTrue(set(fold["stage1_train"]).isdisjoint(fold["stage2_oof"]))

    def test_skillcorner_never_supervises_dead_stacker(self):
        real = fixture("real")
        sc = fixture("sc", source="skillcorner-current", dead=False)
        captured = []
        class Capture:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y, sample_weight=None):
                captured.append((len(y), np.unique(y), sample_weight.copy()))
                return self
        items = []
        for m, weight in ((real, 1.0), (sc, 0.25)):
            items.append((np.full(m.n, .2), np.full(m.n, .6),
                          np.zeros((m.n, 40)), np.zeros((m.n, 65)),
                          np.zeros((m.n, 65)), m, weight))
        with patch.object(pipeline_v4, "EnsembleBinaryClf", Capture):
            pipeline_v4.fit_temporal_stacker_from_oof(items, stride=1)
        valid_real = sum(b - a for a, b in real.blocks)
        live_real = np.sum((real.gt == H) | (real.gt == A))
        live_sc = np.sum((sc.gt == H) | (sc.gt == A))
        self.assertEqual(captured[0][0], valid_real)
        self.assertEqual(captured[1][0], 2 * (live_real + live_sc))
        self.assertTrue(np.any(captured[1][2] < 1))

    def test_unified_metrica_speed_uses_corrupted_tracks(self):
        m = fixture(source="metrica sample-data")
        a = external.degrade(m, seed=5, tracking_version="unified_v4")
        b = external.degrade(dataclasses.replace(m, spd=m.spd + 100), seed=5,
                             tracking_version="unified_v4")
        np.testing.assert_array_equal(a.pos, b.pos)
        np.testing.assert_array_equal(a.spd, b.spd)
        np.testing.assert_array_equal(a.spd, datasets.compute_speed(a.pos, a.blocks))
        self.assertGreaterEqual(a.pos.shape[1], 46)
        self.assertEqual(a.meta["tracking_version"], "unified_v4")
        legacy = external.degrade(m, seed=5)
        self.assertEqual(legacy.pos.shape[1], m.pos.shape[1])

    def test_p3_prefix_symmetry_and_ball_free_columns(self):
        m = fixture()
        swapped = dataclasses.replace(m, team=1 - m.team)
        vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
        rb = ball_tracker.reconstruct_ball(m, vel=vel)
        old = features_v4.build_team_features(m, rb=rb, vel=vel)
        p3 = features_v4.build_team_features(m, rb=rb, vel=vel, use_p3=True)
        np.testing.assert_array_equal(old, p3[:, :old.shape[1]])
        self.assertEqual(p3.shape[1] - old.shape[1], 55)
        direct = features_v4.compute_centroid_axis_motion(swapped, vel=vel)
        mirrored = features_v4.compute_centroid_axis_motion(m, vel=vel, swap=True)
        np.testing.assert_array_equal(direct, mirrored)
        no_ball = dataclasses.replace(m, ball=np.full_like(m.ball, np.nan))
        np.testing.assert_array_equal(direct, features_v4.compute_centroid_axis_motion(
            dataclasses.replace(no_ball, team=1 - m.team), vel=vel))
        cols = features_v4.ball_free_team_cols(p3.shape[1])
        np.testing.assert_array_equal(p3[:, cols],
                                      features_v4.compute_centroid_axis_motion(m, vel=vel))
        np.testing.assert_array_equal(features_v4.build_dead_features(m, vel=vel),
                                      features_v4.build_dead_features(no_ball, vel=vel))

    def test_fully_hidden_block_has_nonzero_gap_length(self):
        m = fixture()
        m.ball[:] = np.nan
        rb = ball_tracker.reconstruct_ball(m)
        for a, b in m.blocks:
            np.testing.assert_array_equal(rb.gap_len[a:b], np.full(b - a, b - a))

    def test_stoppage_augmentation_keeps_labels_players_and_blocks(self):
        m = fixture()
        before_ball, before_gt = m.ball.copy(), m.gt.copy()
        a = datasets.augment_stoppage_ball(m, seed=33)
        b = datasets.augment_stoppage_ball(m, seed=33)
        np.testing.assert_array_equal(a.ball, b.ball)
        np.testing.assert_array_equal(m.ball, before_ball)
        np.testing.assert_array_equal(a.gt, before_gt)
        self.assertIs(a.pos, m.pos)
        self.assertEqual(a.blocks, m.blocks)
        with self.assertRaises(ValueError):
            datasets.augment_stoppage_ball(dataclasses.replace(m, gt=None))


if __name__ == "__main__":
    unittest.main()
