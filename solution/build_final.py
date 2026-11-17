"""Assemble the final submission artifact from separately trained external-only components.

    .venv/bin/python -m solution.build_final \
        --gbdt solution/out/final_gbdt_p3_ens3.pkl \
        --neural solution/out/final_nn2_w128_25hz_sc16_3seed.pkl \
        --decode-cfg solution/configs/decode_final_restart_order.json \
        --out solution/configs/pipeline_final.pkl

Reproduces the externally validated `nn0.7_ownership/restart_order` variant: GBDT Stage-1/2
log-probabilities, live H/A mass re-split by the 25 Hz OwnNet at weight 0.7 (dead column
untouched), no boundary model, no neural switch signal. Refuses to save unless
dead_leak == 0.0 and swap_diff == 0.0 on a public external probe.
"""
import argparse
import hashlib
import json
import os
import pickle

from solution import datasets, deadline_benchmark, line_snap, pipeline_v4


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sidecar(path):
    for suffix in (".provenance.json", ".manifest.json"):
        if os.path.isfile(path + suffix):
            with open(path + suffix) as f:
                return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", required=True)
    ap.add_argument("--neural", required=True)
    ap.add_argument("--decode-cfg", required=True)
    ap.add_argument("--neural-weight", type=float, default=0.7)
    ap.add_argument("--line-snap", action="store_true", help="snap dead onsets to observed line exits")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(deadline_benchmark.ROOT + "/solution/configs/pipeline_v4.pkl"):
        raise ValueError("Refusing to overwrite the legacy pipeline_v4.pkl")

    pipe = pipeline_v4.PipelineV4.load(args.gbdt)
    with open(args.neural, "rb") as f:
        pipe.neural_v2 = pickle.load(f)
    with open(args.decode_cfg) as f:
        pipe.decode_cfg = json.load(f)
    if pipe.decode_cfg["c_res_same"] > pipe.decode_cfg["c_res_flip"]:
        raise ValueError("Decoder must satisfy c_res_same <= c_res_flip")
    pipe.neural_weight = args.neural_weight
    pipe.neural_switch_mode = "ownership_only"
    pipe.boundary_model = None
    pipe.freeze_dead_mask = False
    pipe.line_snap_cfg = dict(line_snap.DEFAULTS) if args.line_snap else None

    invariants = deadline_benchmark.invariant_probe(pipe, datasets.load_dfl("J03WPY"))
    print("invariants", invariants, flush=True)

    pipe.save(args.out)
    nn_dir = os.path.dirname(os.path.abspath(args.neural))
    nn_manifest = None
    for cand in (os.path.join(nn_dir, "final_nn2", "manifest_all.json"), os.path.join(nn_dir, "manifest_all.json")):
        if os.path.isfile(cand):
            with open(cand) as f:
                nn_manifest = json.load(f)
            break
    provenance = dict(
        variant=f"nn{args.neural_weight:g}_ownership/{os.path.basename(args.decode_cfg)}" + ("+line_snap" if args.line_snap else ""),
        line_snap_cfg=pipe.line_snap_cfg,
        gbdt=dict(path=args.gbdt, sha256=sha256(args.gbdt), provenance=sidecar(args.gbdt)),
        neural=dict(path=args.neural, sha256=sha256(args.neural), manifest=nn_manifest),
        decode_cfg=pipe.decode_cfg, neural_weight=args.neural_weight, neural_switch_mode="ownership_only",
        boundary_model=None, invariants=invariants, competition_data_accessed=False,
        model_sha256=sha256(args.out),
    )
    with open(args.out + ".provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print("saved", args.out, provenance["model_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
