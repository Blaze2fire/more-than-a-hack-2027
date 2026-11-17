"""Synthetic checks for the external panel's actual sampling units."""
import copy
import unittest

from solution import summarize_deadline as summary


def row(game, provider, score, *, seed=1, variant=summary.BASELINE_VARIANT,
        scenario="natural"):
    return dict(game=game, provider=provider, scenario=scenario, seed=seed,
                variant=variant, spell_f1=score, spell_f1_merged=score + .01,
                macro_f1=score, alive_dead_f1=score, team_acc=score,
                short_2s_rec=score, burst_strict_team_acc=score)


class AggregationTests(unittest.TestCase):
    def test_masks_collapse_before_equal_provider_mean(self):
        rows = [row("d1", "DFL", .2, seed=i) for i in range(20)]
        rows += [row("d2", "DFL", .4), row("g", "Groningen", .9)]
        report = summary.build_report({"base": rows}, draws=100)
        panel = report["panels"][0]
        self.assertAlmostEqual(panel["overall"]["metrics"]["spell_f1"], .6)
        self.assertEqual(panel["overall"]["n_matches"], 3)
        self.assertEqual(panel["overall"]["n_providers"], 2)

    def test_duplicate_same_mask_does_not_change_sampling_or_interval(self):
        rows = [row("d1", "DFL", .2), row("d2", "DFL", .4), row("g", "Groningen", .9)]
        candidate = [row("d1", "DFL", .3, variant=summary.PRIMARY_VARIANT),
                     row("d2", "DFL", .3, variant=summary.PRIMARY_VARIANT),
                     row("g", "Groningen", .95, variant=summary.PRIMARY_VARIANT)]
        first = summary.build_report({"base": rows, "candidate": candidate}, draws=200)
        second = summary.build_report({"base": rows * 9, "candidate": candidate * 5}, draws=200)
        self.assertEqual(first["primary"], second["primary"])
        self.assertEqual(second["primary"]["paired_vs_base"]["summary"]["n_matches"], 3)
        self.assertEqual(second["duplicate_masks_removed"]["base"], 24)

    def test_scenarios_equal_weight_within_fixture_and_paired_worst_match(self):
        rows = [row("d1", "DFL", .2), row("d2", "DFL", .4), row("g", "Groningen", .9)]
        candidate = [row("d1", "DFL", .3, variant=summary.PRIMARY_VARIANT),
                     row("d2", "DFL", .3, variant=summary.PRIMARY_VARIANT),
                     row("g", "Groningen", .95, variant=summary.PRIMARY_VARIANT)]
        report = summary.build_report({"base": rows, "candidate": candidate}, draws=200)
        paired = report["primary"]["paired_vs_base"]
        self.assertAlmostEqual(paired["summary"]["metrics"]["spell_f1"], .025)
        self.assertAlmostEqual(paired["worst_match"]["spell_f1"]["delta"], -.1)
        self.assertEqual(paired["worst_match"]["spell_f1"]["game"], "d2")
        cases = summary.collapse_masks(summary.normalize_rows([
            row("d1", "DFL", .2, seed=1), row("d1", "DFL", .4, seed=2),
            row("d1", "DFL", .9, scenario="atd_real")], "base")[0])
        self.assertAlmostEqual(summary.collapse_scenarios(cases)[0]["metrics"]["spell_f1"], .6)

    def test_missing_primary_never_selects_highest_scoring_variant(self):
        rows = [row("d1", "DFL", .2), row("d1", "DFL", .99, variant="exploratory")]
        report = summary.build_report({"base": rows}, draws=20)
        self.assertIsNone(report["primary"])
        self.assertIn("no substitute", report["primary_status"])

    def test_different_masks_flag_incomplete_pairing(self):
        rows = [row("d1", "DFL", .2, seed=1)]
        candidate = [row("d1", "DFL", .3, seed=2, variant=summary.PRIMARY_VARIANT)]
        report = summary.build_report({"base": rows, "candidate": candidate}, draws=20)
        paired = report["primary"]["paired_vs_base"]
        self.assertFalse(paired["coverage_complete"])
        self.assertEqual(paired["n_paired_masks"], 0)
        self.assertIsNone(paired["summary"]["metrics"]["spell_f1"])

    def test_conflicting_duplicate_is_rejected(self):
        first = row("d1", "DFL", .2)
        second = copy.deepcopy(first)
        second["spell_f1"] = .3
        with self.assertRaises(ValueError):
            summary.build_report({"base": [first, second]}, draws=20)


if __name__ == "__main__":
    unittest.main()
