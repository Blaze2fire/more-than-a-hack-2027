"""Multi-dataset benchmark & error-separation diagnostic harness (v4).

Evaluates any learned pipeline on:
  - Sportec DFL v1 (7-fold LOMO or zero-shot) across:
      natural, atd_real, atd_blind, atd_carried_dead
    plus the 2-match comparison slice (J03WQQ + J03WR9) matching the repo survey.
  - WCSF Groningen transfer match.
  - Metrica v3 benchmark (held-out LOMO via solution.bench for continuity).

Also reports oracle-dead vs oracle-team separation and frame-level error rates
across visible ball, short gaps (<= 25f), and burst gaps (> 25f).
"""
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from solution import bench, datasets
from solution.common import A, D, H, X, Match


def merge_uncontrolled_gt(m: Match, max_frames: int = 35) -> np.ndarray:
    """Return copy of m.gt with <= max_frames opponent deflections between same-team runs merged."""
    assert m.gt is not None
    out = m.gt.copy()
    for b0, b1 in m.blocks:
        seg = m.gt[b0:b1]
        if len(seg) == 0:
            continue
        chg = np.where(seg[1:] != seg[:-1])[0] + 1
        starts = np.concatenate([[0], chg])
        ends = np.concatenate([chg, [len(seg)]])
        vals = seg[starts]
        lens = ends - starts
        for k in range(1, len(vals) - 1):
            cur = int(vals[k])
            prv = int(vals[k - 1])
            nxt = int(vals[k + 1])
            if cur in (H, A) and prv in (H, A) and prv == nxt and prv != cur and int(lens[k]) <= max_frames:
                out[b0 + int(starts[k]) : b0 + int(ends[k])] = prv
    return out


def score_match(
    name: str,
    scenario: str,
    pred: np.ndarray,
    gt: np.ndarray,
    gt_merged: Optional[np.ndarray] = None,
) -> dict:
    r = pm.evaluate(gt, pred)
    n_gt = max(1, int(r["n_gt_spells"]))
    sf1 = float(r["spell_f1_mean"])
    if gt_merged is not None:
        rm = pm.evaluate(gt_merged, pred)
        sf1_m = float(rm["spell_f1_mean"])
        sf1_obj = 0.5 * (sf1 + sf1_m)
    else:
        sf1_m = sf1
        sf1_obj = sf1
    return dict(
        game=name,
        scenario=scenario,
        spell_f1=sf1,
        spell_f1_merged=sf1_m,
        spell_f1_obj=sf1_obj,
        macro_f1=float(r["macro_f1"]),
        alive_dead_f1=float(r["alive_dead_f1"]),
        team_acc=float(r["team_acc_both_alive"]),
        poss_ratio=float(r["n_pred_spells"] / n_gt),
        n_pred_spells=int(r["n_pred_spells"]),
        n_gt_spells=int(r["n_gt_spells"]),
        share_err=float(r["share_err"]),
    )


def gap_lengths(vis: np.ndarray, blocks: List[Tuple[int, int]]) -> np.ndarray:
    """Return length of current missing-ball burst for every frame (0 if ball visible)."""
    gl = np.zeros(len(vis), dtype=np.int32)
    for b0, b1 in blocks:
        miss = ~vis[b0:b1]
        if not miss.any():
            continue
        padded = np.concatenate([[False], miss, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0]
        for s, e in zip(starts, ends):
            gl[b0 + s : b0 + e] = e - s
    return gl


def oracle_breakdown(
    m: Match,
    logp: np.ndarray,
    decode_fn: Callable[[np.ndarray, List[Tuple[int, int]]], np.ndarray],
) -> dict:
    """Compute ours, oracle-dead, oracle-team, and visibility-regime frame breakdown."""
    gt = m.gt
    graded = np.zeros(m.n, dtype=bool)
    for a, b in m.blocks:
        graded[a:b] = True
    graded &= (gt != X)

    pred = decode_fn(logp, m.blocks)
    ours = pm.evaluate(gt, pred)

    # Oracle-dead: force D wherever gt == D; decode strictly H vs A elsewhere.
    lp_live = logp.copy()
    lp_live[:, D] = -1e6
    team_pred = decode_fn(lp_live, m.blocks)
    od = np.where(gt == D, D, team_pred)
    od[~graded] = X
    r_od = pm.evaluate(gt, od)

    # Oracle-team: keep our alive/dead mask, substitute gt team where both are live.
    ot = pred.copy()
    both = graded & (pred != D) & (gt != D)
    ot[both] = gt[both]
    ot[~graded] = X
    r_ot = pm.evaluate(gt, ot)

    # Breakdown by visible / short gap (1..25f) / burst gap (>25f)
    vis = np.isfinite(m.ball[:, 0])
    gl = gap_lengths(vis, m.blocks)
    regimes = {
        "vis": graded & vis,
        "short": graded & (~vis) & (gl <= 25),
        "burst": graded & (~vis) & (gl > 25),
    }
    reg_stats = {}
    for rk, mask in regimes.items():
        if mask.sum() == 0:
            reg_stats[rk] = dict(pct=0.0, acc=np.nan, ad_acc=np.nan, team_acc=np.nan)
            continue
        live_both = mask & (pred != D) & (gt != D)
        reg_stats[rk] = dict(
            pct=float(mask.sum() / graded.sum()),
            acc=float(np.mean(pred[mask] == gt[mask])),
            ad_acc=float(np.mean((pred[mask] == D) == (gt[mask] == D))),
            team_acc=float(np.mean(pred[live_both] == gt[live_both])) if live_both.any() else np.nan,
        )

    return dict(
        ours_spell_f1=float(ours["spell_f1_mean"]),
        oracle_dead_spell_f1=float(r_od["spell_f1_mean"]),
        oracle_team_spell_f1=float(r_ot["spell_f1_mean"]),
        ours_macro_f1=float(ours["macro_f1"]),
        ours_poss_ratio=float(ours["n_pred_spells"] / max(1, ours["n_gt_spells"])),
        regimes=reg_stats,
    )


def eval_zero_shot_legacy(logp_fn, decode_fn, scenarios=("natural", "atd_real", "atd_blind", "atd_carried_dead"), seed_base=100):
    """Evaluate a pre-trained scorer across DFL (all 7 matches) & Groningen without retraining."""
    raw_dfl = datasets.load_all_dfl()
    gron = datasets.load_groningen()
    rows = []
    diags = []
    for sc in scenarios:
        for i, m_raw in enumerate(raw_dfl):
            m = datasets.degrade_atd(m_raw, scenario=sc, seed=seed_base + i)
            lp = logp_fn(m)
            pred = decode_fn(lp, m.blocks)
            rows.append(score_match(m.name, sc, pred, m.gt, gt_merged=merge_uncontrolled_gt(m)))
            if sc in ("natural", "atd_real"):
                diags.append(oracle_breakdown(m, lp, decode_fn) | {"game": m.name, "scenario": sc})
        # Also evaluate Groningen under natural & atd_real
        if sc in ("natural", "atd_real"):
            gm = datasets.degrade_atd(gron, scenario=sc, seed=seed_base + 99)
            lp = logp_fn(gm)
            pred = decode_fn(lp, gm.blocks)
            rows.append(score_match(gm.name, sc, pred, gm.gt, gt_merged=merge_uncontrolled_gt(gm)))
    return pd.DataFrame(rows), pd.DataFrame(diags)


def print_summary(df: pd.DataFrame, title: str = "Benchmark Summary") -> None:
    print(f"\n=== {title} ===")
    cols = [
        c for c in ("spell_f1", "spell_f1_merged", "spell_f1_obj", "macro_f1", "alive_dead_f1", "team_acc", "poss_ratio")
        if c in df.columns
    ]
    dfl = df[df["game"].str.startswith("dfl_")]
    if not dfl.empty:
        grp = dfl.groupby("scenario", sort=False)[cols].mean()
        print("DFL 7-match mean:")
        print(grp.round(4).to_string())
        w2 = dfl[dfl["game"].isin(["dfl_J03WQQ", "dfl_J03WR9"])]
        if not w2.empty:
            print("DFL J03WQQ+J03WR9 slice:")
            wcols = [c for c in ("spell_f1", "spell_f1_merged", "spell_f1_obj", "macro_f1", "poss_ratio") if c in df.columns]
            print(w2.groupby("scenario", sort=False)[wcols].mean().round(4).to_string())
    gr = df[df["game"] == "groningen"]
    if not gr.empty:
        print("Groningen transfer:")
        print(gr[["scenario"] + cols].round(4).to_string(index=False))

