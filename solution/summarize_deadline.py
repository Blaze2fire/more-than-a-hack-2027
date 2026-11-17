"""Summarize external deadline panels without treating mask seeds as matches.

Example:
    python -m solution.summarize_deadline --base solution/out/control/results.json \
        --candidate solution/out/candidate/results.json --out-prefix docs/DEADLINE_RESULTS

Only aggregate rows are read. The fixed primary candidate is declared below;
this module never selects a winner from the evaluated panel.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BASELINE_VARIANT = "gbdt/baseline"
PRIMARY_VARIANT = "nn0.7_ownership/frozen_dead"
METRICS = ("spell_f1", "spell_f1_merged", "macro_f1", "alive_dead_f1",
           "team_acc", "short_2s_rec", "burst_strict_team_acc")
EXPECTED_SCENARIOS = ("natural", "atd_real", "atd_blind", "atd_carried_dead",
                      "gap-stress-burst-01")
CAVEATS = [
    "Fixtures have historical development exposure; this is not an untouched final test.",
    "Mask repetitions are averaged within fixture/scenario and never counted as independent matches.",
    "Scenarios are averaged within each fixture; fixtures within each provider; providers receive equal weight.",
    "Intervals resample whole matches within each fixed provider. They condition on the observed providers and do not estimate uncertainty across new providers.",
    "Singleton Groningen contributes no within-provider bootstrap variation; only three providers and few fixtures limit uncertainty estimates.",
    "Native/raw and merged labels are distinct endpoints. Merged results do not replace native/raw SpellF1.",
    "All non-primary variants are descriptive ablations. No all-panel winner or multiple-comparison-adjusted superiority is asserted.",
    "team_acc is conditional on both truth and prediction being alive; burst_strict_team_acc counts dead/unknown predictions as ownership errors.",
]


def _mean(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(finite)) if finite else None


def _metric_values(row):
    result = {}
    for metric in METRICS:
        value = row.get(metric)
        if value is None:
            result[metric] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric aggregate {metric}")
        else:
            result[metric] = float(value) if math.isfinite(value) else None
    if result["spell_f1"] is None:
        raise ValueError("Every aggregate row requires finite native/raw spell_f1")
    return result


def normalize_rows(rows, source):
    """Deduplicate the same mask; conflicting copies are an invalid input."""
    unique, duplicates = {}, 0
    for row in rows:
        for key in ("game", "provider", "scenario", "variant", "seed"):
            if key not in row:
                raise ValueError(f"Missing aggregate row key: {key}")
        identity = tuple(str(row[k]) for k in ("variant", "provider", "game", "scenario", "seed"))
        values = _metric_values(row)
        if identity in unique:
            if unique[identity]["metrics"] != values:
                raise ValueError("Conflicting results for the same source/variant/fixture/scenario/seed")
            duplicates += 1
            continue
        unique[identity] = dict(source=source, variant=identity[0], provider=identity[1],
                                game=identity[2], scenario=identity[3], seed=identity[4],
                                metrics=values)
    return list(unique.values()), duplicates


def collapse_masks(rows):
    """First average repeated masks, retaining their unique count for inspection."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["variant"], row["provider"], row["game"], row["scenario"])].append(row)
    result = []
    for (source, variant, provider, game, scenario), group in sorted(groups.items()):
        result.append(dict(source=source, variant=variant, provider=provider, game=game,
                           scenario=scenario, n_masks=len(group),
                           metrics={m: _mean(r["metrics"][m] for r in group) for m in METRICS}))
    return result


def collapse_scenarios(rows):
    """A fixture remains one sampling unit, irrespective of masks or scenarios."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["provider"], row["game"])].append(row)
    return [dict(provider=provider, game=game,
                 scenarios=sorted(r["scenario"] for r in group),
                 metrics={m: _mean(r["metrics"][m] for r in group) for m in METRICS})
            for (provider, game), group in sorted(groups.items())]


def provider_balanced(fixtures):
    groups = defaultdict(list)
    for row in fixtures:
        groups[row["provider"]].append(row)
    provider_rows = []
    for provider, group in sorted(groups.items()):
        provider_rows.append(dict(provider=provider, n_matches=len(group),
                                  metrics={m: _mean(r["metrics"][m] for r in group) for m in METRICS}))
    return dict(n_matches=len(fixtures), n_providers=len(groups),
                metrics={m: _mean(r["metrics"][m] for r in provider_rows) for m in METRICS},
                metric_n_matches={m: sum(r["metrics"][m] is not None for r in fixtures) for m in METRICS},
                metric_n_providers={m: sum(r["metrics"][m] is not None for r in provider_rows) for m in METRICS},
                providers=provider_rows)


def _bootstrap(fixtures, draws, seed):
    """Paired match-block resampling, stratified by the fixed provider panel."""
    rng = np.random.default_rng(seed)
    providers = sorted({r["provider"] for r in fixtures})
    samples, contributors = {m: [] for m in METRICS}, {m: 0 for m in METRICS}
    for provider in providers:
        group = [r for r in fixtures if r["provider"] == provider]
        indices = rng.integers(0, len(group), size=(draws, len(group)))
        for metric in METRICS:
            values = np.array([np.nan if r["metrics"][metric] is None else r["metrics"][metric]
                               for r in group], dtype=float)
            if not np.isfinite(values).any():
                continue
            selected = values[indices]
            counts = np.isfinite(selected).sum(axis=1)
            means = np.divide(np.nansum(selected, axis=1), counts,
                              out=np.full(draws, np.nan), where=counts > 0)
            samples[metric].append(means)
            contributors[metric] += 1
    result = {}
    for metric in METRICS:
        if not samples[metric]:
            result[metric] = None
            continue
        matrix = np.stack(samples[metric])
        # Require every contributing provider in a draw rather than silently
        # changing provider weights when an optional metric lacks observations.
        distribution = matrix.mean(axis=0)
        distribution = distribution[np.isfinite(distribution)]
        result[metric] = (dict(lower=float(np.quantile(distribution, .025)),
                               upper=float(np.quantile(distribution, .975)),
                               valid_draws=len(distribution),
                               contributing_providers=contributors[metric])
                          if len(distribution) else None)
    return result


def paired_comparison(baseline_rows, candidate_rows, draws=5000, seed=20261117):
    """Pair identical fixture/scenario/mask identities before collapsing masks."""
    def index(rows):
        return {(r["provider"], r["game"], r["scenario"], r["seed"]): r for r in rows}
    baseline, candidate = index(baseline_rows), index(candidate_rows)
    shared = sorted(baseline.keys() & candidate.keys())
    case_groups = defaultdict(list)
    for key in shared:
        left, right = baseline[key], candidate[key]
        delta = {m: (right["metrics"][m] - left["metrics"][m]
                      if right["metrics"][m] is not None and left["metrics"][m] is not None else None)
                 for m in METRICS}
        case_groups[key[:3]].append(delta)
    cases = [dict(provider=p, game=g, scenario=s, n_masks=len(group),
                  metrics={m: _mean(r[m] for r in group) for m in METRICS})
             for (p, g, s), group in sorted(case_groups.items())]
    fixtures = collapse_scenarios(cases)
    summary = provider_balanced(fixtures)
    intervals = _bootstrap(fixtures, draws, seed) if fixtures else {m: None for m in METRICS}
    worst = {}
    for metric in METRICS:
        eligible = [r for r in fixtures if r["metrics"][metric] is not None]
        item = min(eligible, key=lambda r: r["metrics"][metric]) if eligible else None
        worst[metric] = (dict(provider=item["provider"], game=item["game"], delta=item["metrics"][metric])
                         if item else None)
    return dict(summary=summary, bootstrap_95=intervals, worst_match=worst,
                n_paired_masks=len(shared), n_paired_fixture_scenarios=len(cases),
                coverage_complete=baseline.keys() == candidate.keys(),
                baseline_only_masks=[dict(provider=p, game=g, scenario=s, seed=z)
                                     for p, g, s, z in sorted(baseline.keys() - candidate.keys())],
                candidate_only_masks=[dict(provider=p, game=g, scenario=s, seed=z)
                                      for p, g, s, z in sorted(candidate.keys() - baseline.keys())],
                fixture_deltas=fixtures, fixture_scenario_deltas=cases)


def build_report(inputs, draws=5000, seed=20261117):
    """Create summaries from ``{base: rows, candidate: rows}`` aggregate inputs."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    all_rows, duplicate_counts = [], {}
    for source, rows in inputs.items():
        normalized, duplicate_counts[source] = normalize_rows(rows, source)
        all_rows.extend(normalized)
    baseline = [r for r in all_rows if r["source"] == "base" and r["variant"] == BASELINE_VARIANT]
    if not baseline:
        raise ValueError(f"--base must contain {BASELINE_VARIANT}")
    primary_source = "candidate" if "candidate" in inputs else "base"
    panel_rows = collapse_masks(all_rows)
    panels = []
    for source, variant in sorted({(r["source"], r["variant"]) for r in all_rows}):
        raw_group = [r for r in all_rows if (r["source"], r["variant"]) == (source, variant)]
        cases = [r for r in panel_rows if (r["source"], r["variant"]) == (source, variant)]
        scenarios = sorted({r["scenario"] for r in cases})
        panel = dict(source=source, variant=variant,
                     predeclared_primary=(source == primary_source and variant == PRIMARY_VARIANT),
                     overall=provider_balanced(collapse_scenarios(cases)), scenarios=[])
        for scenario in scenarios:
            selected = [r for r in cases if r["scenario"] == scenario]
            panel["scenarios"].append(dict(scenario=scenario,
                                            **provider_balanced(collapse_scenarios(selected))))
        panel["paired_vs_base"] = paired_comparison(baseline, raw_group, draws, seed)
        panels.append(panel)
    primary = next((p for p in panels if p["predeclared_primary"]), None)
    baseline_scenarios = sorted({r["scenario"] for r in baseline})
    scenarios_by_fixture = defaultdict(set)
    for row in baseline:
        scenarios_by_fixture[(row["provider"], row["game"])].add(row["scenario"])
    return dict(schema_version=1,
                protocol=dict(baseline_source="base", baseline_variant=BASELINE_VARIANT,
                              primary_source=primary_source, primary_variant=PRIMARY_VARIANT,
                              primary_endpoint="provider-balanced native/raw SpellF1; equal scenarios per fixture",
                              baseline_scenarios=baseline_scenarios,
                              expected_scenarios=list(EXPECTED_SCENARIOS),
                              expected_scenario_panel_complete=all(
                                  set(EXPECTED_SCENARIOS) <= scenarios
                                  for scenarios in scenarios_by_fixture.values()),
                              bootstrap_draws=draws, bootstrap_seed=seed,
                              interval="95% percentile; match blocks within fixed providers",
                              selection="Fixed primary; all other panels descriptive; no automatic promotion"),
                caveats=CAVEATS, duplicate_masks_removed=duplicate_counts,
                primary_status="available" if primary else "not evaluated; no substitute selected",
                primary=primary, panels=panels, fixture_scenario_rows=panel_rows)


def _fmt(value):
    return "—" if value is None else f"{value:.4f}"


def markdown_report(report):
    protocol = report["protocol"]
    lines = ["# Deadline external benchmark results", "",
             f"Predeclared primary: **{protocol['primary_source']} / {PRIMARY_VARIANT}**. "
             f"Reference: **base / {BASELINE_VARIANT}**. No panel winner is selected.", "",
             "Mask seeds are averaged within fixture/scenario, scenarios within fixture, "
             "matches within provider, and providers equally. Raw/native SpellF1 is the "
             "primary endpoint; merged labels are a separate sensitivity analysis.", "",
             f"Primary status: {report['primary_status']}.", "",
             "## Provider-balanced panel", "",
             "| Source | Variant | Matches | Providers | Raw SpellF1 | Merged SpellF1 | MacroF1 | Paired raw Δ | 95% match-bootstrap interval | Worst match Δ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for panel in report["panels"]:
        overall = panel["overall"]
        paired = panel["paired_vs_base"]
        interval = paired["bootstrap_95"]["spell_f1"]
        worst = paired["worst_match"]["spell_f1"]
        ci = f"[{interval['lower']:.4f}, {interval['upper']:.4f}]" if interval else "—"
        variant = panel["variant"] + (" **(primary)**" if panel["predeclared_primary"] else "")
        lines.append(f"| {panel['source']} | {variant} | {overall['n_matches']} | {overall['n_providers']} | "
                     f"{_fmt(overall['metrics']['spell_f1'])} | {_fmt(overall['metrics']['spell_f1_merged'])} | "
                     f"{_fmt(overall['metrics']['macro_f1'])} | {_fmt(paired['summary']['metrics']['spell_f1'])} | "
                     f"{ci} | {_fmt(worst['delta'] if worst else None)} |")
    lines.extend(["", "## All provider and scenario panels", "",
                  "| Source | Variant | Provider | Scenario | Matches | Raw | Merged | MacroF1 | Alive/dead F1 | Conditional team accuracy | ≤2 s recall | Strict burst team accuracy |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for panel in report["panels"]:
        for scenario in panel["scenarios"]:
            for provider in scenario["providers"]:
                metric = provider["metrics"]
                lines.append(f"| {panel['source']} | {panel['variant']} | {provider['provider']} | "
                             f"{scenario['scenario']} | {provider['n_matches']} | " +
                             " | ".join(_fmt(metric[m]) for m in METRICS) + " |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend("- " + caveat for caveat in report["caveats"])
    if not protocol["expected_scenario_panel_complete"]:
        lines.append("- The expected five-scenario baseline panel is incomplete; inspect coverage before interpreting the aggregate.")
    for panel in report["panels"]:
        if not panel["paired_vs_base"]["coverage_complete"]:
            lines.append(f"- {panel['source']} / {panel['variant']} has different mask coverage from the baseline; paired deltas use only identical shared masks.")
    lines.extend(["", "The JSON companion contains per-fixture deltas, every aggregated fixture/scenario, "
                  "metric coverage counts, input hashes, all paired intervals and unpaired mask identities.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="External deadline_benchmark results.json")
    parser.add_argument("--candidate", help="Optional second pipeline's external results.json")
    parser.add_argument("--out-prefix", default="docs/DEADLINE_RESULTS")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    inputs, identities = {}, {}
    for source, filename in (("base", args.base), ("candidate", args.candidate)):
        if filename is None:
            continue
        payload = Path(filename).read_bytes()
        data = json.loads(payload)
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise ValueError("Expected a results.json object with aggregate rows")
        inputs[source] = data["rows"]
        identities[source] = dict(filename=Path(filename).name,
                                   sha256=hashlib.sha256(payload).hexdigest(),
                                   model_sha256=data.get("protocol", {}).get("model_sha256"))
    report = build_report(inputs, draws=args.bootstrap, seed=args.seed)
    report["inputs"] = identities
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Saved external aggregate reports: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
