"""Predeclared conditional-ownership confirmation on the final four SkillCorner fixtures.

No parameters are fitted or selected. SkillCorner has no native dead labels, so
only ownership accuracy, Brier score and NLL on known live H/A frames are scored.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from solution import datasets, neural_v2
from solution.common import A, H

CONFIRM_IDS = ("2013725", "2015213", "2016236", "2017461")
SEEDS = (911, 2027)


def protocol():
    return dict(
        fixtures=list(CONFIRM_IDS), selection="Final four IDs in sorted 20-match SkillCorner Current inventory",
        views=["natural/911", "atd_real/911", "atd_real/2027"],
        metrics=["conditional_team_accuracy", "brier", "nll"],
        labels="Provider native known live H/A only; no dead-ball metric",
        purpose="Confirmation only; do not tune on these scores",
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def evaluate(model_paths, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    registered = out / "protocol.json"
    if not registered.exists():
        raise ValueError("Register the protocol before inspecting any confirmation scores")
    p = json.loads(registered.read_text())
    if p["fixtures"] != list(CONFIRM_IDS):
        raise ValueError("Confirmation fixture register changed")
    if tuple(datasets.list_skillcorner_current()[-4:]) != CONFIRM_IDS:
        raise ValueError("SkillCorner inventory changed; retain the predeclared fixture list")
    models, manifests = {}, {}
    for name, path in model_paths.items():
        path = Path(path)
        mp = path.with_name(path.name.replace("models_", "manifest_").replace(".pkl", ".json"))
        manifest = json.loads(mp.read_text())
        if manifest["train_keys"] != ["dfl0", "dfl1", "dfl2", "dfl3"]:
            raise ValueError("Confirmation requires both fixed first-four DFL neural models")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError("Neural checkpoint checksum mismatch")
        with path.open("rb") as f:
            models[name] = pickle.load(f)
        manifests[name] = manifest
    rows = []
    for mid in CONFIRM_IDS:
        raw = datasets.load_skillcorner_current(mid)
        if raw.meta.get("has_real_dead_ball", True):
            raise ValueError("Unexpected native dead labels in SkillCorner")
        for scenario in ("natural", "atd_real"):
            for seed in (SEEDS[:1] if scenario == "natural" else SEEDS):
                m = datasets.degrade_atd(raw, scenario=scenario, seed=seed)
                mask = np.zeros(m.n, dtype=bool)
                for a, b in m.blocks:
                    mask[a:b] = True
                mask &= (m.gt == H) | (m.gt == A)
                y = (m.gt[mask] == H).astype(np.float64)
                for name, ensemble in models.items():
                    ph, _ = neural_v2.predict_match(ensemble, m)
                    ph = np.clip(ph[mask].astype(np.float64), 1e-6, 1 - 1e-6)
                    row = dict(model=name, fixture=mid, scenario=scenario, seed=seed, n_live=int(mask.sum()),
                               conditional_team_accuracy=float(np.mean((ph >= 0.5) == y)),
                               brier=float(np.mean((ph - y) ** 2)),
                               nll=float(np.mean(-y * np.log(ph) - (1 - y) * np.log1p(-ph))))
                    rows.append(row)
                (out / "results.json").write_text(json.dumps(dict(protocol=p, models=manifests, rows=rows), indent=2))
                print(f"Scored external SkillCorner {mid}/{scenario}/{seed}", flush=True)
    # Match/view means: this avoids giving a longer fixture extra weight.
    summary = {}
    for name in models:
        summary[name] = {}
        for scenario in ("natural", "atd_real"):
            selected = [r for r in rows if r["model"] == name and r["scenario"] == scenario]
            summary[name][scenario] = {key: float(np.mean([r[key] for r in selected]))
                                      for key in ("conditional_team_accuracy", "brier", "nll")}
    (out / "results.json").write_text(json.dumps(dict(protocol=p, models=manifests, rows=rows, summary=summary), indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--mixed", default="solution/out/deadline_neural/models_4_5_6_g.pkl")
    parser.add_argument("--short", default="solution/out/deadline_neural_short/models_4_5_6_g.pkl")
    parser.add_argument("--out-dir", default="solution/out/deadline_neural_confirm")
    args = parser.parse_args()
    if args.register:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / "protocol.json"
        if target.exists():
            raise ValueError("Confirmation protocol already registered; it cannot be silently replaced")
        target.write_text(json.dumps(protocol(), indent=2))
        print("Registered confirmation fixtures before scoring:", ", ".join(CONFIRM_IDS))
        return
    neural_v2.torch.set_num_threads(2)
    evaluate(dict(mixed=args.mixed, short=args.short), args.out_dir)


if __name__ == "__main__":
    main()
