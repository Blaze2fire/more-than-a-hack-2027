"""Prevent correctness checks from becoming a public-GT model-selection channel."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from solution import sanity


class PublicDataGuardrails(unittest.TestCase):
    def test_external_model_ancestry_rejects_hidden_test_or_imputer_leakage(self):
        from solution import deadline_benchmark
        names = ["dfl_" + mid for mid in deadline_benchmark.TRAIN_IDS]
        fixtures = [{"name": name, "source": "sportec-dfl-v1"} for name in names]
        actual = {"fixtures": fixtures, "inner_folds": [
            {"stage1_train": names[1:], "imputer_train": names[1:], "stage2_oof": names[:1]}]}
        manifest = dict(actual, training_skillcorner=[])
        pipe = SimpleNamespace(training_provenance=actual)
        self.assertEqual(deadline_benchmark.validate_training_provenance(manifest, pipe), set(names))
        actual["inner_folds"][0]["imputer_train"] = names
        with self.assertRaisesRegex(ValueError, "ancestry"):
            deadline_benchmark.validate_training_provenance(manifest, pipe)
        actual["inner_folds"][0]["imputer_train"] = names[1:]
        actual["fixtures"].append({"name": "groningen", "source": "wcsf-groningen"})
        with self.assertRaisesRegex(ValueError, "Unexpected fitted"):
            deadline_benchmark.validate_training_provenance(manifest, pipe)

    def test_sanity_reads_only_alignment_inputs_and_never_scores(self):
        truth = np.r_[np.zeros(160), np.ones(200), np.full(180, 2)].astype(np.int8)
        reads = []
        def read(path):
            reads.append(path)
            return truth.copy()
        output = io.StringIO()
        with patch.object(sanity.sys, "argv", ["sanity", "candidate.csv"]), \
             patch.object(sanity.pm, "read_label_csv", read), \
             patch.object(sanity.pm, "label_codes", lambda value: value), \
             patch.object(sanity.pm, "evaluate", side_effect=AssertionError("Public scoring forbidden")), \
             contextlib.redirect_stdout(output):
            sanity.main()
        self.assertEqual(reads, ["candidate.csv", sanity.PUBLIC_GT])
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertNotIn("non_regression", output.getvalue())
        self.assertNotIn("score", output.getvalue())

    def test_retired_visualizer_fails_before_loading_any_asset(self):
        from solution import visualize_5min
        with patch.object(visualize_5min.atd, "load", side_effect=AssertionError("Asset must not be read")):
            with self.assertRaisesRegex(SystemExit, "Retired"):
                visualize_5min.main()

    def test_gt_video_viewer_takes_no_prediction_input(self):
        # Organizer email: checking our output against the video is not allowed.
        from solution import gt_video_viewer
        with contextlib.redirect_stderr(io.StringIO()):
            for flag in ("--pred", "--csv", "--model", "--pipeline", "--baseline-csv"):
                with self.assertRaises(SystemExit):
                    gt_video_viewer.main([flag, "x"])

    def test_gt_video_viewer_run_length_encoding(self):
        from solution import gt_video_viewer
        labels = np.array(list("XXHHHDAX"), dtype=object)
        self.assertEqual(gt_video_viewer.runs(labels),
                         [[0, 1, "X"], [2, 4, "H"], [5, 5, "D"], [6, 6, "A"], [7, 7, "X"]])
        page = gt_video_viewer.build_html(gt_video_viewer.runs(labels), 8, "v.mp4")
        self.assertIn('"video": "v.mp4"', page)
        self.assertNotIn("/*__DATA__*/null", page)


if __name__ == "__main__":
    unittest.main()
