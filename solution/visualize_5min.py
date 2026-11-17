"""RETIRED: public-GT failure inspection violates the solution's data contract.

The historical implementation is retained for provenance; main exits before
loading any asset. Use external fixtures for failure analysis.

Reads local BAR-ATH tracking + public 5-min GT + model stage outputs, computes full
evaluation metrics via `evaluation/possession_metrics.py`, and writes a self-contained
HTML file for local browser inspection without printing raw competition frames to stdout.

Usage:
    .venv/bin/python -m solution.visualize_5min \
        --model solution/configs/pipeline_v4.pkl \
        --baseline-csv evaluation/results/more_than_a_hack_2027_e014_bak.csv
"""
import argparse
import html
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from solution import atd, bench_v4, pipeline_v4, viterbi4
from solution.common import A, D, FPS, H, Match, ROOT, X

if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_MODEL = os.path.join(ROOT, "solution", "configs", "pipeline_v4.pkl")
DEFAULT_BASELINE_CSV = os.path.join(ROOT, "evaluation", "results", "more_than_a_hack_2027_e014_bak.csv")
DEFAULT_VITERBI_CFG = os.path.join(ROOT, "solution", "configs", "viterbi4.json")


def _compute_metrics_bundle(
    gt_full: np.ndarray,
    pred_full: np.ndarray,
    lp_full: Optional[np.ndarray],
    m: Match,
    decode_cfg: dict,
    sw_sig: Optional[np.ndarray],
    w_start: int,
    w_end: int,
) -> dict:
    """Evaluate all official possession_metrics + short-spell + oracle + visibility regimes on 5-min window."""
    gt_w = gt_full[w_start:w_end].copy()
    pred_w = pred_full[w_start:w_end].copy()
    pred_w[gt_w == X] = X

    ev = pm.evaluate(gt_w, pred_w, taus=(6, 12, 25, 50), bootstrap_n=0)

    runs = pm.valid_runs(gt_w)
    filled = pm.fill_abstentions(pred_w, runs)
    refs = pm.possession_spells(gt_w, runs)
    preds = pm.possession_spells(filled, runs)
    durs = np.array([e - s for s, e, _ in refs], dtype=np.int32)
    matched = pm.matched_gt_flags(pm.spell_overlaps(refs, preds), len(refs), 0.5)
    m2 = durs <= 50
    m1 = durs <= 25
    ev["short_2s_recall"] = float(matched[m2].mean()) if m2.any() else None
    ev["short_2s_count"] = f"{int(matched[m2].sum())}/{int(m2.sum())}"
    ev["sub1s_recall"] = float(matched[m1].mean()) if m1.any() else None
    ev["sub1s_count"] = f"{int(matched[m1].sum())}/{int(m1.sum())}"

    # Strip non-serializable pair tuples inside transitions for clean JSON embedding
    clean_trans = {}
    for tol, td in ev["transitions"].items():
        clean_trans[str(tol)] = {
            k: v for k, v in td.items() if k not in ("pairs", "missed", "spurious")
        }
    ev["transitions"] = clean_trans
    ev["spell_f1"] = {str(k): v for k, v in ev["spell_f1"].items()}

    # Oracle & regime breakdown inside covered 5-min GT window
    graded = gt_w != X
    vis = np.isfinite(m.ball[w_start:w_end, 0])
    gl = bench_v4.gap_lengths(np.isfinite(m.ball[:, 0]), m.blocks)[w_start:w_end]
    regimes = {}
    for rk, rmask in [
        ("vis", graded & vis),
        ("short", graded & (~vis) & (gl <= 25)),
        ("burst", graded & (~vis) & (gl > 25)),
    ]:
        if rmask.sum() == 0:
            regimes[rk] = {"pct": 0.0, "acc": None, "ad_acc": None, "team_acc": None, "frames": 0}
        else:
            lb = rmask & (pred_w != D) & (gt_w != D)
            regimes[rk] = {
                "pct": float(rmask.sum() / max(1, graded.sum())),
                "acc": float(np.mean(pred_w[rmask] == gt_w[rmask])),
                "ad_acc": float(np.mean((pred_w[rmask] == D) == (gt_w[rmask] == D))),
                "team_acc": float(np.mean(pred_w[lb] == gt_w[lb])) if lb.any() else None,
                "frames": int(rmask.sum()),
            }
    ev["regimes"] = regimes

    if lp_full is not None:
        m_win = Match(
            name=m.name,
            ball=m.ball,
            pos=m.pos,
            spd=m.spd,
            team=m.team,
            blocks=[(w_start, w_end)],
            pitch=m.pitch,
            gt=gt_full,
        )
        diag = bench_v4.oracle_breakdown(
            m_win,
            lp_full,
            lambda lpx, bl: viterbi4.decode4(lpx, bl, decode_cfg, sw_signal=sw_sig),
        )
        ev["oracle_dead_spell_f1"] = float(diag["oracle_dead_spell_f1"])
        ev["oracle_team_spell_f1"] = float(diag["oracle_team_spell_f1"])
    else:
        ev["oracle_dead_spell_f1"] = None
        ev["oracle_team_spell_f1"] = None

    return ev


def _build_inspectors(
    gt_w: np.ndarray,
    pred_w: np.ndarray,
    vis_w: np.ndarray,
    w_start: int,
) -> Tuple[List[int], List[dict], List[dict]]:
    """Build per-frame failure code strip, GT spell table, and transition latency table."""
    n_w = len(gt_w)
    # 0=OK/X, 1=FalseDead, 2=MissedDead, 3=WrongTeam(Vis), 4=WrongTeam(Gap)
    err_codes = np.zeros(n_w, dtype=np.int8)
    g_ok = gt_w != X
    err_codes[g_ok & ((gt_w == H) | (gt_w == A)) & (pred_w == D)] = 1
    err_codes[g_ok & (gt_w == D) & ((pred_w == H) | (pred_w == A))] = 2
    wrong_t = g_ok & ((gt_w == H) | (gt_w == A)) & ((pred_w == H) | (pred_w == A)) & (pred_w != gt_w)
    err_codes[wrong_t & vis_w] = 3
    err_codes[wrong_t & (~vis_w)] = 4

    lbl_map = {H: "H", A: "A", D: "D", X: "X"}
    runs = pm.valid_runs(gt_w)
    filled = pm.fill_abstentions(pred_w, runs)
    gt_spells = pm.possession_spells(gt_w, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    # Best IoU per GT spell & one-to-one matched flag at IoU >= 0.5
    best_iou = [0.0] * len(gt_spells)
    for iou, gi, _ in overlaps:
        if iou > best_iou[gi]:
            best_iou[gi] = float(iou)
    matched_05 = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)

    spell_rows = []
    for i, (s, e, c) in enumerate(gt_spells):
        dur_s = (e - s) / FPS
        vis_pct = 100.0 * float(vis_w[s:e].mean()) if e > s else 0.0
        seg_err = err_codes[s:e]
        n_fd = int((seg_err == 1).sum())
        n_wt_v = int((seg_err == 3).sum())
        n_wt_g = int((seg_err == 4).sum())
        if matched_05[i]:
            diag = "Matched (IoU>=0.5)"
        elif n_fd > 0.45 * (e - s):
            diag = "False Dead suppression"
        elif n_wt_g >= n_wt_v and n_wt_g > 0:
            diag = "Wrong team in burst gap"
        elif n_wt_v > 0:
            diag = "Wrong team (visible ball / split)"
        elif (e - s) <= 50:
            diag = "Short <=2s spell absorbed/shifted"
        else:
            diag = "Boundary shift / fragmentation"
        spell_rows.append({
            "idx": i + 1,
            "team": lbl_map[c],
            "rel_s": int(s),
            "rel_e": int(e),
            "abs_s": int(w_start + s),
            "abs_e": int(w_start + e),
            "dur_s": round(dur_s, 2),
            "vis_pct": round(vis_pct, 1),
            "best_iou": round(best_iou[i], 3),
            "matched_05": bool(matched_05[i]),
            "diag": diag,
        })

    gt_tr = pm.transitions(gt_w, runs)
    pr_tr = pm.transitions(filled, runs)
    ts25 = pm.transition_scores(gt_tr, pr_tr, tol=25, typed="to")
    tr_rows = []
    for g_t, p_t in ts25["pairs"]:
        tr_rows.append({
            "rel_f": int(g_t[0]),
            "abs_f": int(w_start + g_t[0]),
            "transition": f"{lbl_map[g_t[1]]}->{lbl_map[g_t[2]]}",
            "pred_transition": f"{lbl_map[p_t[1]]}->{lbl_map[p_t[2]]}",
            "status": "matched",
            "lat_f": int(p_t[0] - g_t[0]),
        })
    for g_t in ts25["missed"]:
        tr_rows.append({
            "rel_f": int(g_t[0]),
            "abs_f": int(w_start + g_t[0]),
            "transition": f"{lbl_map[g_t[1]]}->{lbl_map[g_t[2]]}",
            "pred_transition": "-",
            "status": "missed",
            "lat_f": None,
        })
    for p_t in ts25["spurious"]:
        tr_rows.append({
            "rel_f": int(p_t[0]),
            "abs_f": int(w_start + p_t[0]),
            "transition": "-",
            "pred_transition": f"{lbl_map[p_t[1]]}->{lbl_map[p_t[2]]}",
            "status": "spurious",
            "lat_f": None,
        })
    tr_rows.sort(key=lambda r: r["rel_f"])
    return err_codes.tolist(), spell_rows, tr_rows


def _render_html(payload_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BAR-ATH 5-Min Public GT Live Visualizer & Failure Debugger</title>
<style>
  :root {{
    --bg: #0b0f17;
    --panel: #131a26;
    --panel2: #182132;
    --border: #263248;
    --text: #e6edf7;
    --muted: #8ea0b8;
    --home: #ff4d5a;
    --away: #3fa2ff;
    --dead: #8c97a8;
    --gold: #ffbe3b;
    --green: #38d399;
    --purple: #b17dff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 14px; background: var(--bg); color: var(--text);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; line-height: 1.4;
  }}
  .header {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 12px; gap: 10px;
  }}
  .title h1 {{ margin: 0; font-size: 17px; letter-spacing: -0.01em; }}
  .title .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  .kpi-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 10px; margin-bottom: 12px;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
  }}
  .kpi .lbl {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .kpi .val {{ font-size: 20px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }}
  .kpi .delta {{ font-size: 11px; margin-top: 2px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--home); }}
  .neu {{ color: var(--muted); }}
  .workspace {{
    display: grid; grid-template-columns: minmax(420px, 46%) 1fr; gap: 12px; margin-bottom: 12px;
  }}
  @media (max-width: 1100px) {{ .workspace {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
  }}
  .card h2 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }}
  canvas {{ display: block; width: 100%; border-radius: 6px; background: #0d1612; }}
  #timelineCanvas {{ background: #0e1420; cursor: crosshair; }}
  .controls {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px;
  }}
  button, select {{
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    padding: 5px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }}
  button:hover {{ border-color: var(--away); }}
  button.active {{ background: #223554; border-color: var(--away); }}
  .frame-info {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); font-size: 12px;
  }}
  .pill {{
    display: inline-block; padding: 2px 7px; border-radius: 999px; font-weight: 600; font-size: 11px;
  }}
  .p-H {{ background: rgba(255,77,90,0.2); color: var(--home); border: 1px solid rgba(255,77,90,0.45); }}
  .p-A {{ background: rgba(63,162,255,0.2); color: var(--away); border: 1px solid rgba(63,162,255,0.45); }}
  .p-D {{ background: rgba(140,151,168,0.2); color: var(--dead); border: 1px solid rgba(140,151,168,0.45); }}
  .p-vis {{ background: rgba(56,211,153,0.18); color: var(--green); }}
  .p-gap {{ background: rgba(255,190,59,0.2); color: var(--gold); }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  tr.click-row:hover {{ background: rgba(63,162,255,0.12); cursor: pointer; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--muted); margin-top: 6px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  .scroll-tbl {{ max-height: 360px; overflow-y: auto; }}
  .err-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }}
  .chip {{ padding: 4px 8px; border-radius: 6px; font-size: 11px; background: var(--panel2); border: 1px solid var(--border); }}
</style>
</head>
<body>

<div class="header">
  <div class="title">
    <h1>BAR-ATH 5-Min Public GT Live Visualizer & Failure-Mode Debugger</h1>
    <div class="sub" id="metaSub">Loading...</div>
  </div>
  <div class="err-chips" id="errSummaryChips"></div>
</div>

<div class="kpi-row" id="kpiRow"></div>

<div class="workspace">
  <div class="card">
    <h2>
      <span>25 fps Pitch Replay & Ball Reconstruction</span>
      <span id="frameBadge" class="neu">Frame -</span>
    </h2>
    <canvas id="pitchCanvas" width="760" height="500"></canvas>
    <div class="legend">
      <span><i class="dot" style="background:var(--home)"></i>Home (H) + Centroid</span>
      <span><i class="dot" style="background:var(--away)"></i>Away (A) + Centroid</span>
      <span><i class="dot" style="background:#fff;border:1px solid #000"></i>Raw ATD Ball</span>
      <span><i class="dot" style="background:var(--gold)"></i>Reconstructed ball_hat (radius=1-conf)</span>
    </div>
    <div class="controls">
      <button id="btnPlay">Play (Space)</button>
      <button id="btnStepB">-1f</button>
      <button id="btnStepF">+1f</button>
      <button id="btnJumpB">-1s</button>
      <button id="btnJumpF">+1s</button>
      <select id="speedSel">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1.0x</option>
        <option value="2">2.0x</option>
        <option value="4">4.0x</option>
      </select>
      <button id="btnPrevErr">Prev Error</button>
      <button id="btnNextErr">Next Error</button>
      <button id="btnNextTrans">Next GT Trans</button>
    </div>
    <div class="frame-info" id="frameInfoGrid"></div>
  </div>

  <div class="card">
    <h2>
      <span>Synchronized Multi-Stage Timeline & Probability Streams</span>
      <span style="display:flex;gap:8px;align-items:center;font-weight:400;font-size:12px">
        Zoom:
        <select id="zoomSel">
          <option value="1" selected>Full 5-min (1x)</option>
          <option value="3">3x (~100s)</option>
          <option value="8">8x (~38s)</option>
          <option value="20">20x (~15s)</option>
        </select>
        <label><input type="checkbox" id="chkShowNN" checked /> NN</label>
        <label><input type="checkbox" id="chkShowBF" checked /> BF</label>
        <label><input type="checkbox" id="chkShowSW" checked /> Switch</label>
      </span>
    </h2>
    <canvas id="timelineCanvas" width="900" height="495"></canvas>
    <div style="margin-top:8px">
      <input type="range" id="scrubber" min="0" max="100" value="0" style="width:100%" />
    </div>
    <div class="legend">
      <span><i class="dot" style="background:var(--home)"></i>H</span>
      <span><i class="dot" style="background:var(--away)"></i>A</span>
      <span><i class="dot" style="background:var(--dead)"></i>D</span>
      <span>| Curves: <b style="color:var(--green)">P(H|live) S2</b>, <b style="color:var(--purple)">Neural</b>, <b style="color:var(--gold)">BF-Spec</b>, <b style="color:#ff7e5f">S1-GBDT</b>, <b style="color:#94a3b8">P(D) S2</b>, <b style="color:#38bdf8">SwitchSig</b></span>
    </div>
  </div>
</div>

<div class="card">
  <div class="tabs">
    <button class="tab-btn active" data-tab="tabMetrics">All Official Metrics, Oracles, Regimes & Confusion Matrix</button>
    <button class="tab-btn" data-tab="tabSpells">GT Spells Inspector (Click to Seek)</button>
    <button class="tab-btn" data-tab="tabTransitions">Transitions & Latency Inspector (tol=25)</button>
  </div>
  <div id="tabMetrics" class="tab-pane"></div>
  <div id="tabSpells" class="tab-pane" style="display:none"></div>
  <div id="tabTransitions" class="tab-pane" style="display:none"></div>
</div>

<script id="payloadData" type="application/json">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("payloadData").textContent);
const N = DATA.n_win;
const FPS = DATA.fps;
const LAB = ["H", "A", "D", "X"];
const COL = {{ 0: "#ff4d5a", 1: "#3fa2ff", 2: "#7c889b", 3: "#2b3342" }};
const ERR_NAMES = ["Match / Ungraded", "False Dead (GT Live -> D)", "Missed Dead (GT D -> Live)", "Wrong Team (Visible Ball)", "Wrong Team (Missing-Ball Gap)"];
const ERR_COLS = ["transparent", "#ffbe3b", "#e879f9", "#ef4444", "#f97316"];

let curF = 0;
let playing = false;
let lastTs = 0;
let accum = 0;

document.getElementById("metaSub").textContent =
  `${{DATA.model_tag}} vs E014 Baseline | Window abs frames ${{DATA.w_start}}..${{DATA.w_end - 1}} (${{(N / FPS).toFixed(1)}}s, ${{N}} frames @ ${{FPS}} Hz)`;

function fmtPct(v) {{ return v == null ? "-" : (v * 100).toFixed(2) + "%"; }}
function fmtNum(v, d=4) {{ return v == null ? "-" : Number(v).toFixed(d); }}

function buildHeaderAndMetrics() {{
  const nm = DATA.metrics_new;
  const bm = DATA.metrics_base;

  // Error summary chips
  const counts = [0, 0, 0, 0, 0];
  for (let i = 0; i < N; i++) counts[DATA.err_codes[i]]++;
  const chipEl = document.getElementById("errSummaryChips");
  [1, 2, 3, 4].forEach(c => {{
    const span = document.createElement("span");
    span.className = "chip";
    span.style.borderColor = ERR_COLS[c];
    span.innerHTML = `<b style="color:${{ERR_COLS[c]}}">${{ERR_NAMES[c]}}</b>: ${{counts[c]}}f (${{(counts[c]/FPS).toFixed(1)}}s)`;
    chipEl.appendChild(span);
  }});

  // KPI cards
  const kpis = [
    ["SpellF1 (Mean IoU)", nm.spell_f1_mean, bm.spell_f1_mean, 4],
    ["SpellF1 @ IoU 0.5", nm.spell_f1_at_headline, bm.spell_f1_at_headline, 4],
    ["Macro-F1", nm.macro_f1, bm.macro_f1, 4],
    ["Alive/Dead F1", nm.alive_dead_f1, bm.alive_dead_f1, 4],
    ["Team Acc (Both Alive)", nm.team_acc_both_alive, bm.team_acc_both_alive, 4],
    ["Pred / GT Spells", nm.n_pred_spells, bm.n_pred_spells, 0, nm.n_gt_spells],
    ["Trans F1 (tol=12)", nm.transitions["12"].f1, bm.transitions["12"].f1, 4],
    ["Trans F1 (tol=25)", nm.transitions["25"].f1, bm.transitions["25"].f1, 4],
  ];
  const rowEl = document.getElementById("kpiRow");
  kpis.forEach(([title, nv, bv, dec, extraGt]) => {{
    const div = document.createElement("div");
    div.className = "kpi";
    const diff = nv - bv;
    const dCls = Math.abs(diff) < 1e-6 ? "neu" : (diff > 0 ? "pos" : "neg");
    const dTxt = dec === 0 ? `vs ${{bv}} (GT: ${{extraGt}})` : `${{diff >= 0 ? "+" : ""}}${{diff.toFixed(dec)}} vs E014 (${{bv.toFixed(dec)}})`;
    div.innerHTML = `<div class="lbl">${{title}}</div><div class="val">${{dec === 0 ? nv : nv.toFixed(dec)}}</div><div class="delta ${{dCls}}">${{dTxt}}</div>`;
    rowEl.appendChild(div);
  }});

  // Detailed metrics tab
  const tMet = document.getElementById("tabMetrics");
  let rowsHtml = "";
  const cmpRows = [
    ["spell_f1_mean", nm.spell_f1_mean, bm.spell_f1_mean],
    ["spell_f1 @ IoU 0.3", nm.spell_f1["0.3"].f1, bm.spell_f1["0.3"].f1],
    ["spell_f1 @ IoU 0.4", nm.spell_f1["0.4"].f1, bm.spell_f1["0.4"].f1],
    ["spell_f1 @ IoU 0.5", nm.spell_f1["0.5"].f1, bm.spell_f1["0.5"].f1],
    ["spell_f1 @ IoU 0.6", nm.spell_f1["0.6"].f1, bm.spell_f1["0.6"].f1],
    ["spell_f1 @ IoU 0.7", nm.spell_f1["0.7"].f1, bm.spell_f1["0.7"].f1],
    ["macro_f1", nm.macro_f1, bm.macro_f1],
    ["accuracy", nm.accuracy, bm.accuracy],
    ["cohens_kappa", nm.kappa, bm.kappa],
    ["alive_dead_f1", nm.alive_dead_f1, bm.alive_dead_f1],
    ["team_acc_gt_alive", nm.team_acc_gt_alive, bm.team_acc_gt_alive],
    ["team_acc_both_alive", nm.team_acc_both_alive, bm.team_acc_both_alive],
    ["short_2s_recall @ 0.5", nm.short_2s_recall, bm.short_2s_recall],
    ["sub1s_recall @ 0.5", nm.sub1s_recall, bm.sub1s_recall],
    ["mean_spell_duration_s (GT " + nm.gt_mean_spell_duration_s.toFixed(2) + "s)", nm.mean_spell_duration_s, bm.mean_spell_duration_s],
    ["share_err (GT H " + (nm.gt_home_share*100).toFixed(1) + "%)", nm.share_err, bm.share_err],
    ["fragmentation", nm.fragmentation, bm.fragmentation],
    ["oracle_dead_spell_f1", nm.oracle_dead_spell_f1, null],
    ["oracle_team_spell_f1", nm.oracle_team_spell_f1, null],
  ];
  cmpRows.forEach(([k, nv, bv]) => {{
    const d = (nv != null && bv != null) ? (nv - bv) : null;
    const dStr = d == null ? "-" : ((d >= 0 ? "+" : "") + d.toFixed(4));
    const cls = d == null ? "neu" : (k.includes("share_err") ? (d <= 0 ? "pos" : "neg") : (d >= 0 ? "pos" : "neg"));
    rowsHtml += `<tr><td>${{k}}</td><td><b>${{fmtNum(nv)}}</b></td><td>${{fmtNum(bv)}}</td><td class="${{cls}}">${{dStr}}</td></tr>`;
  }});

  let clsHtml = "";
  ["H", "A", "D"].forEach(c => {{
    const nc = nm.per_class[c], bc = bm.per_class[c];
    clsHtml += `<tr><td><b>${{c}}</b> (${{nc.support}}f)</td><td>${{nc.precision.toFixed(4)}} / ${{nc.recall.toFixed(4)}} / <b>${{nc.f1.toFixed(4)}}</b></td><td>${{bc.precision.toFixed(4)}} / ${{bc.recall.toFixed(4)}} / ${{bc.f1.toFixed(4)}}</td></tr>`;
  }});

  let trHtml = "";
  ["6", "12", "25", "50"].forEach(tol => {{
    const nt = nm.transitions[tol], bt = bm.transitions[tol];
    trHtml += `<tr><td>&plusmn;${{tol}}f (${{(tol/FPS).toFixed(2)}}s)</td><td>P=${{nt.precision.toFixed(3)}} R=${{nt.recall.toFixed(3)}} <b>F1=${{nt.f1.toFixed(3)}}</b> | med=${{nt.median_latency==null?'-':nt.median_latency.toFixed(1)}}f, p90=${{nt.p90_abs_latency==null?'-':nt.p90_abs_latency.toFixed(1)}}f</td><td>P=${{bt.precision.toFixed(3)}} R=${{bt.recall.toFixed(3)}} F1=${{bt.f1.toFixed(3)}}</td></tr>`;
  }});

  let regHtml = "";
  ["vis", "short", "burst"].forEach(rk => {{
    const nr = nm.regimes[rk], br = bm.regimes[rk];
    regHtml += `<tr><td><b>${{rk}}</b> (${{(nr.pct*100).toFixed(1)}}%, ${{nr.frames}}f)</td><td>acc=${{fmtPct(nr.acc)}} | AD=${{fmtPct(nr.ad_acc)}} | Team=${{fmtPct(nr.team_acc)}}</td><td>acc=${{fmtPct(br.acc)}} | AD=${{fmtPct(br.ad_acc)}} | Team=${{fmtPct(br.team_acc)}}</td></tr>`;
  }});

  const cm = nm.confusion;
  let cmHtml = `<table><tr><th>GT \\\\ Pred</th><th>H</th><th>A</th><th>D</th><th>X</th></tr>`;
  ["H", "A", "D"].forEach((lbl, r) => {{
    cmHtml += `<tr><td><b>${{lbl}}</b></td><td>${{cm[r][0]}}</td><td>${{cm[r][1]}}</td><td>${{cm[r][2]}}</td><td>${{cm[r][3]}}</td></tr>`;
  }});
  cmHtml += `</table>`;

  tMet.innerHTML = `<div class="grid-2">
    <div>
      <h3>Headline, Spell IoU & Oracles</h3>
      <table><tr><th>Metric</th><th>New Model</th><th>E014 Base</th><th>Delta</th></tr>${{rowsHtml}}</table>
    </div>
    <div>
      <h3>Per-Class (P / R / F1)</h3>
      <table><tr><th>Class</th><th>New</th><th>E014</th></tr>${{clsHtml}}</table>
      <h3 style="margin-top:14px">Transitions & Latency</h3>
      <table><tr><th>Tolerance</th><th>New Model</th><th>E014</th></tr>${{trHtml}}</table>
      <h3 style="margin-top:14px">Ball-Visibility Regimes</h3>
      <table><tr><th>Regime</th><th>New</th><th>E014</th></tr>${{regHtml}}</table>
      <h3 style="margin-top:14px">Confusion Matrix (New Model)</h3>
      ${{cmHtml}}
    </div>
  </div>`;

  // Spells Tab
  const tSp = document.getElementById("tabSpells");
  let spHtml = `<div class="scroll-tbl"><table><tr><th>#</th><th>Team</th><th>Frames (Abs)</th><th>Dur</th><th>Ball Vis</th><th>Best IoU</th><th>Status</th><th>Failure Diagnosis</th></tr>`;
  DATA.spells.forEach(s => {{
    const st = s.matched_05 ? `<span class="pos">Matched</span>` : `<span class="neg">Missed@0.5</span>`;
    spHtml += `<tr class="click-row" onclick="seekFrame(${{s.rel_s}}, 8)"><td>${{s.idx}}</td><td><span class="pill p-${{s.team}}">${{s.team}}</span></td><td>${{s.abs_s}}..${{s.abs_e}}</td><td>${{s.dur_s}}s</td><td>${{s.vis_pct}}%</td><td>${{s.best_iou}}</td><td>${{st}}</td><td>${{s.diag}}</td></tr>`;
  }});
  spHtml += `</table></div>`;
  tSp.innerHTML = spHtml;

  // Transitions Tab
  const tTr = document.getElementById("tabTransitions");
  let txHtml = `<div class="scroll-tbl"><table><tr><th>Abs Frame</th><th>Rel Time</th><th>GT Trans</th><th>Pred Trans</th><th>Status (tol=25f)</th><th>Latency</th></tr>`;
  DATA.transitions.forEach(t => {{
    const cls = t.status === "matched" ? "pos" : "neg";
    const lat = t.lat_f == null ? "-" : `${{t.lat_f >= 0 ? "+" : ""}}${{t.lat_f}}f (${{(t.lat_f/FPS*1000).toFixed(0)}}ms)`;
    txHtml += `<tr class="click-row" onclick="seekFrame(${{t.rel_f}}, 8)"><td>${{t.abs_f}}</td><td>${{(t.rel_f/FPS).toFixed(2)}}s</td><td>${{t.transition}}</td><td>${{t.pred_transition}}</td><td class="${{cls}}">${{t.status}}</td><td>${{lat}}</td></tr>`;
  }});
  txHtml += `</table></div>`;
  tTr.innerHTML = txHtml;
}}

// Interactive tabs
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).style.display = "block";
  }});
}});

function seekFrame(f, optZoom) {{
  curF = Math.max(0, Math.min(N - 1, f));
  if (optZoom && document.getElementById("zoomSel").value === "1") {{
    document.getElementById("zoomSel").value = String(optZoom);
  }}
  renderAll();
}}

// Draw 105.3 x 68 m pitch & frame players/ball
const pCanvas = document.getElementById("pitchCanvas");
const pCtx = pCanvas.getContext("2d");
const L_PITCH = DATA.pitch[0], W_PITCH = DATA.pitch[1];

function toPx(xm, ym) {{
  const pad = 28;
  const uw = pCanvas.width - 2 * pad, uh = pCanvas.height - 2 * pad;
  return [pad + (xm / L_PITCH) * uw, pad + (ym / W_PITCH) * uh];
}}

function drawPitchMarkings() {{
  pCtx.fillStyle = "#0e2017";
  pCtx.fillRect(0, 0, pCanvas.width, pCanvas.height);
  pCtx.strokeStyle = "rgba(255,255,255,0.28)";
  pCtx.lineWidth = 1.5;
  const [x0, y0] = toPx(0, 0), [x1, y1] = toPx(L_PITCH, W_PITCH);
  pCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  const [mx0, my0] = toPx(L_PITCH/2, 0), [mx1, my1] = toPx(L_PITCH/2, W_PITCH);
  pCtx.beginPath(); pCtx.moveTo(mx0, my0); pCtx.lineTo(mx1, my1); pCtx.stroke();
  const [cx, cy] = toPx(L_PITCH/2, W_PITCH/2);
  const r915 = (9.15 / L_PITCH) * (x1 - x0);
  pCtx.beginPath(); pCtx.arc(cx, cy, r915, 0, Math.PI * 2); pCtx.stroke();
  // Penalty areas (16.5m x 40.32m)
  const pyA = W_PITCH/2 - 20.16, pyB = W_PITCH/2 + 20.16;
  const [pl0, pt0] = toPx(0, pyA), [pl1, pt1] = toPx(16.5, pyB);
  pCtx.strokeRect(pl0, pt0, pl1 - pl0, pt1 - pt0);
  const [pr0, pb0] = toPx(L_PITCH - 16.5, pyA), [pr1, pb1] = toPx(L_PITCH, pyB);
  pCtx.strokeRect(pr0, pb0, pr1 - pr0, pb1 - pb0);
}}

function drawPitch(f) {{
  drawPitchMarkings();
  const players = DATA.players_q[f];
  let hx=0, hy=0, hn=0, ax=0, ay=0, an=0;
  for (let s = 0; s < DATA.n_slots; s++) {{
    const xq = players[s*2], yq = players[s*2 + 1];
    if (xq < -100) continue;
    const xm = xq / 10.0, ym = yq / 10.0;
    const isH = DATA.team[s] === 0;
    if (isH) {{ hx += xm; hy += ym; hn++; }} else {{ ax += xm; ay += ym; an++; }}
    const [px, py] = toPx(xm, ym);
    // Short velocity tail from t-5f if valid
    if (f >= 5) {{
      const pxq = DATA.players_q[f - 5][s*2], pyq = DATA.players_q[f - 5][s*2 + 1];
      if (pxq > -100) {{
        const [ox, oy] = toPx(pxq / 10.0, pyq / 10.0);
        pCtx.strokeStyle = isH ? "rgba(255,77,90,0.45)" : "rgba(63,162,255,0.45)";
        pCtx.lineWidth = 2;
        pCtx.beginPath(); pCtx.moveTo(ox, oy); pCtx.lineTo(px, py); pCtx.stroke();
      }}
    }}
    pCtx.fillStyle = isH ? "#ff4d5a" : "#3fa2ff";
    pCtx.beginPath(); pCtx.arc(px, py, 5.5, 0, Math.PI * 2); pCtx.fill();
    pCtx.strokeStyle = "#0b0f17"; pCtx.lineWidth = 1; pCtx.stroke();
  }}
  // Centroid-axis vector (Priority 3 visual overlay!)
  if (hn > 0 && an > 0) {{
    const [chx, chy] = toPx(hx/hn, hy/hn);
    const [cax, cay] = toPx(ax/an, ay/an);
    pCtx.strokeStyle = "rgba(255,255,255,0.22)";
    pCtx.setLineDash([5, 4]);
    pCtx.beginPath(); pCtx.moveTo(chx, chy); pCtx.lineTo(cax, cay); pCtx.stroke();
    pCtx.setLineDash([]);
    pCtx.strokeStyle = "#ff4d5a"; pCtx.lineWidth = 2; pCtx.strokeRect(chx-5, chy-5, 10, 10);
    pCtx.strokeStyle = "#3fa2ff"; pCtx.strokeRect(cax-5, cay-5, 10, 10);
  }}
  // Reconstructed ball + confidence halo
  const bhx = DATA.rb_xy_q[f*2] / 10.0, bhy = DATA.rb_xy_q[f*2 + 1] / 10.0;
  const conf = DATA.p_bconf[f] / 255.0;
  const [bxp, byp] = toPx(bhx, bhy);
  const rad = 6 + (1 - conf) * 26;
  pCtx.strokeStyle = `rgba(255,190,59,${{0.25 + 0.5*conf}})`;
  pCtx.lineWidth = 1.5;
  pCtx.beginPath(); pCtx.arc(bxp, byp, rad, 0, Math.PI * 2); pCtx.stroke();
  pCtx.fillStyle = "#ffbe3b";
  pCtx.beginPath(); pCtx.arc(bxp, byp, 4.5, 0, Math.PI * 2); pCtx.fill();
  // Raw ATD ball ring
  const rbxq = DATA.raw_ball_q[f*2], rbyq = DATA.raw_ball_q[f*2 + 1];
  if (rbxq > -100) {{
    const [rxp, ryp] = toPx(rbxq / 10.0, rbyq / 10.0);
    pCtx.strokeStyle = "#ffffff"; pCtx.lineWidth = 2;
    pCtx.beginPath(); pCtx.arc(rxp, ryp, 6, 0, Math.PI * 2); pCtx.stroke();
  }}
}}

// Draw Synchronized Multi-Stage Timeline
const tCanvas = document.getElementById("timelineCanvas");
const tCtx = tCanvas.getContext("2d");

function getViewBounds() {{
  const z = Number(document.getElementById("zoomSel").value);
  const span = Math.max(120, Math.floor(N / z));
  let s = Math.max(0, curF - Math.floor(span / 2));
  if (s + span > N) s = Math.max(0, N - span);
  return [s, Math.min(N, s + span)];
}}

function drawLabelBar(arr, y, h, s, e, padL, uw) {{
  let i = s;
  while (i < e) {{
    const code = arr[i];
    let j = i + 1;
    while (j < e && arr[j] === code) j++;
    const x0 = padL + ((i - s) / (e - s)) * uw;
    const x1 = padL + ((j - s) / (e - s)) * uw;
    tCtx.fillStyle = COL[code];
    tCtx.fillRect(x0, y, Math.max(1, x1 - x0), h);
    i = j;
  }}
}}

function drawTimeline() {{
  const W = tCanvas.width, Hh = tCanvas.height;
  tCtx.fillStyle = "#0e1420";
  tCtx.fillRect(0, 0, W, Hh);
  const padL = 95, padR = 12, uw = W - padL - padR;
  const [s, e] = getViewBounds();

  const tracks = [
    ["Ground Truth", DATA.gt_w, 12, 22],
    ["New Final", DATA.pred_w, 40, 22],
    ["E014 Baseline", DATA.base_w, 68, 22],
    ["Raw Viterbi4", DATA.vit_raw_w, 96, 20],
  ];
  tCtx.font = "11px Inter, sans-serif";
  tracks.forEach(([name, arr, y, h]) => {{
    tCtx.fillStyle = "#9fb0c7";
    tCtx.fillText(name, 8, y + 15);
    drawLabelBar(arr, y, h, s, e, padL, uw);
  }});

  // Gap & Failure Strip
  tCtx.fillStyle = "#9fb0c7"; tCtx.fillText("Errors / Gaps", 8, 138);
  for (let px = 0; px < uw; px++) {{
    const fi = Math.min(N - 1, s + Math.floor((px / uw) * (e - s)));
    const gap = DATA.rb_gap[fi];
    tCtx.fillStyle = gap === 0 ? "rgba(56,211,153,0.35)" : (gap <= 25 ? "rgba(250,204,21,0.5)" : "rgba(255,190,59,0.85)");
    tCtx.fillRect(padL + px, 123, 1, 7);
    const ec = DATA.err_codes[fi];
    if (ec > 0) {{
      tCtx.fillStyle = ERR_COLS[ec];
      tCtx.fillRect(padL + px, 132, 2, 10);
    }}
  }}

  // Probability Curves Chart area
  const cy0 = 160, cy1 = Hh - 24, ch = cy1 - cy0;
  tCtx.fillStyle = "#121a29"; tCtx.fillRect(padL, cy0, uw, ch);
  tCtx.strokeStyle = "rgba(255,255,255,0.12)";
  [0.25, 0.5, 0.75].forEach(v => {{
    const yy = cy1 - v * ch;
    tCtx.beginPath(); tCtx.moveTo(padL, yy); tCtx.lineTo(padL + uw, yy); tCtx.stroke();
  }});
  tCtx.fillStyle = "#9fb0c7"; tCtx.fillText("Prob Streams", 8, cy0 + 18);
  tCtx.fillText("1.0", padL - 25, cy0 + 8);
  tCtx.fillText("0.5", padL - 25, cy0 + ch/2 + 4);
  tCtx.fillText("0.0", padL - 25, cy1);

  const plotCurve = (arr255, color, width, dash=false) => {{
    if (!arr255 || !arr255.length) return;
    tCtx.strokeStyle = color;
    tCtx.lineWidth = width;
    if (dash) tCtx.setLineDash([4, 3]);
    tCtx.beginPath();
    const step = Math.max(1, Math.floor((e - s) / uw));
    let started = false;
    for (let i = s; i < e; i += step) {{
      const x = padL + ((i - s) / (e - s)) * uw;
      const y = cy1 - (arr255[i] / 255.0) * ch;
      if (!started) {{ tCtx.moveTo(x, y); started = true; }} else tCtx.lineTo(x, y);
    }}
    tCtx.stroke();
    tCtx.setLineDash([]);
  }};

  plotCurve(DATA.p_d2, "#94a3b8", 1.5, true);
  plotCurve(DATA.p_ha1, "#ff7e5f", 1.1);
  if (document.getElementById("chkShowBF").checked) plotCurve(DATA.p_ha_bf, "#ffbe3b", 1.3);
  if (document.getElementById("chkShowNN").checked) plotCurve(DATA.p_ha_nn, "#b17dff", 1.6);
  if (document.getElementById("chkShowSW").checked) plotCurve(DATA.p_sw, "#38bdf8", 1.2);
  plotCurve(DATA.p_h2, "#38d399", 2.0);

  // Current playhead cursor
  const curX = padL + ((curF - s) / Math.max(1, e - s)) * uw;
  tCtx.strokeStyle = "#ffffff";
  tCtx.lineWidth = 1.8;
  tCtx.beginPath(); tCtx.moveTo(curX, 6); tCtx.lineTo(curX, cy1 + 4); tCtx.stroke();
}}

function updateFrameInfo(f) {{
  document.getElementById("frameBadge").textContent = `Abs Frame ${{DATA.w_start + f}} (${{(f/FPS).toFixed(2)}}s / ${{(N/FPS).toFixed(1)}}s)`;
  document.getElementById("scrubber").max = String(N - 1);
  document.getElementById("scrubber").value = String(f);
  const gtL = LAB[DATA.gt_w[f]], prL = LAB[DATA.pred_w[f]], bsL = LAB[DATA.base_w[f]];
  const ec = DATA.err_codes[f];
  const gap = DATA.rb_gap[f];
  const gCls = gap === 0 ? "p-vis" : "p-gap";
  const gTxt = gap === 0 ? "Visible Ball" : `Missing Gap (${{gap}}f / ${{(gap/FPS).toFixed(1)}}s)`;
  const nnP = DATA.p_ha_nn.length ? (DATA.p_ha_nn[f]/255*100).toFixed(1)+"%" : "-";
  const bfP = DATA.p_ha_bf.length ? (DATA.p_ha_bf[f]/255*100).toFixed(1)+"%" : "-";
  document.getElementById("frameInfoGrid").innerHTML = `
    <div><b>GT:</b> <span class="pill p-${{gtL}}">${{gtL}}</span></div>
    <div><b>New:</b> <span class="pill p-${{prL}}">${{prL}}</span></div>
    <div><b>E014:</b> <span class="pill p-${{bsL}}">${{bsL}}</span></div>
    <div><span class="pill ${{gCls}}">${{gTxt}}</span></div>
    <div><b>P(D) S2:</b> ${{(DATA.p_d2[f]/255*100).toFixed(1)}}%</div>
    <div><b>P(H|L) S2:</b> ${{(DATA.p_h2[f]/255*100).toFixed(1)}}%</div>
    <div><b>NN P(H):</b> ${{nnP}} | <b>BF:</b> ${{bfP}}</div>
    <div><b>Diag:</b> <span style="color:${{ec?ERR_COLS[ec]:'#38d399'}}">${{ERR_NAMES[ec]}}</span></div>
  `;
}}

function renderAll() {{
  drawPitch(curF);
  drawTimeline();
  updateFrameInfo(curF);
}}

// Playback & Scrubber events
document.getElementById("btnPlay").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("btnPlay").textContent = playing ? "Pause (Space)" : "Play (Space)";
  lastTs = performance.now();
}});
document.getElementById("btnStepB").addEventListener("click", () => {{ playing = false; curF = Math.max(0, curF - 1); renderAll(); }});
document.getElementById("btnStepF").addEventListener("click", () => {{ playing = false; curF = Math.min(N - 1, curF + 1); renderAll(); }});
document.getElementById("btnJumpB").addEventListener("click", () => {{ curF = Math.max(0, curF - FPS); renderAll(); }});
document.getElementById("btnJumpF").addEventListener("click", () => {{ curF = Math.min(N - 1, curF + FPS); renderAll(); }});
document.getElementById("scrubber").addEventListener("input", ev => {{ curF = Number(ev.target.value); renderAll(); }});
["zoomSel", "chkShowNN", "chkShowBF", "chkShowSW"].forEach(id => document.getElementById(id).addEventListener("change", renderAll));

tCanvas.addEventListener("click", ev => {{
  const rect = tCanvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (tCanvas.width / rect.width);
  const padL = 95, padR = 12, uw = tCanvas.width - padL - padR;
  if (x >= padL && x <= padL + uw) {{
    const [s, e] = getViewBounds();
    curF = Math.max(0, Math.min(N - 1, s + Math.round(((x - padL) / uw) * (e - s))));
    renderAll();
  }}
}});

document.getElementById("btnNextErr").addEventListener("click", () => {{
  let i = curF + 1;
  while (i < N && (DATA.err_codes[i] === 0 || DATA.err_codes[i - 1] === DATA.err_codes[i])) i++;
  if (i >= N) {{ i = 0; while (i < curF && DATA.err_codes[i] === 0) i++; }}
  curF = Math.min(N - 1, i); renderAll();
}});
document.getElementById("btnPrevErr").addEventListener("click", () => {{
  let i = Math.max(0, curF - 1);
  while (i > 0 && DATA.err_codes[i] === DATA.err_codes[curF]) i--;
  while (i > 0 && DATA.err_codes[i] === 0) i--;
  curF = i; renderAll();
}});
document.getElementById("btnNextTrans").addEventListener("click", () => {{
  const hit = DATA.transitions.find(t => t.rel_f > curF && t.transition !== "-");
  curF = hit ? hit.rel_f : (DATA.transitions[0]?.rel_f || 0);
  renderAll();
}});

window.addEventListener("keydown", ev => {{
  if (ev.code === "Space") {{ ev.preventDefault(); document.getElementById("btnPlay").click(); }}
  else if (ev.code === "ArrowRight") {{ curF = Math.min(N - 1, curF + (ev.shiftKey ? 25 : 1)); renderAll(); }}
  else if (ev.code === "ArrowLeft") {{ curF = Math.max(0, curF - (ev.shiftKey ? 25 : 1)); renderAll(); }}
}});

function tick(ts) {{
  if (playing) {{
    const dt = (ts - lastTs) / 1000.0;
    lastTs = ts;
    accum += dt * FPS * Number(document.getElementById("speedSel").value);
    if (accum >= 1.0) {{
      const step = Math.floor(accum);
      accum -= step;
      curF += step;
      if (curF >= N) curF = 0;
      renderAll();
    }}
  }}
  requestAnimationFrame(tick);
}}

buildHeaderAndMetrics();
renderAll();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main():
    raise SystemExit(
        "Retired by docs/DATA_CONTRACT.md: public-GT failure inspection and variant "
        "comparison are prohibited. Use solution.deadline_benchmark on external fixtures."
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-tag", default="PipelineV4-Final")
    p.add_argument("--baseline-csv", default=DEFAULT_BASELINE_CSV)
    p.add_argument("--decode-cfg", default=DEFAULT_VITERBI_CFG)
    p.add_argument("--out-html", default=os.path.join(ROOT, "docs", "live_visualizer_5min.html"))
    p.add_argument("--copy-report", default=os.path.join(ROOT, "evaluation", "reports", "live_visualizer_5min.html"))
    args = p.parse_args()

    m = atd.load()
    gt_full = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt_full)
    w_start, w_end = int(runs[0][0]), int(runs[-1][1])

    pipe = pipeline_v4.PipelineV4.load(args.model)
    if os.path.isfile(args.decode_cfg):
        with open(args.decode_cfg) as f:
            pipe.decode_cfg = json.load(f)

    rb, vel, pd1, pha1, pha_bf, pha_nn, psw_nn, lp, sw_sig = pipe.predict_components(m)
    pred_full = viterbi4.decode4(lp, m.blocks, pipe.decode_cfg, sw_signal=sw_sig)

    # Pre-repair raw 4-state Viterbi sequence (to show exact spell repair edits)
    vit_raw = np.full(len(lp), X, dtype=np.int8)
    dec_p = dict(viterbi4.DECODE4_DEFAULTS, **(pipe.decode_cfg or {}))
    c_sw = float(dec_p.get("c_sw_gain", 0.0))
    s_scaled = float(dec_p["w"]) * lp
    for b0, b1 in m.blocks:
        blk_sw = None
        if c_sw > 0.0:
            step_sig = viterbi4._symmetric_prob_changepoint(lp[b0:b1], win=10)
            blk_sw = (0.55 * sw_sig[b0:b1] + 0.45 * step_sig) if sw_sig is not None else step_sig
        vit_raw[b0:b1] = viterbi4.viterbi4_block(
            s_scaled[b0:b1, H],
            s_scaled[b0:b1, A],
            s_scaled[b0:b1, D],
            c_team=float(dec_p["c_team"]),
            c_stop=float(dec_p["c_stop"]),
            c_res_same=float(dec_p["c_res_same"]),
            c_res_flip=float(dec_p["c_res_flip"]),
            d_bias=float(dec_p["d_bias"]),
            sw_signal=blk_sw,
            c_sw_gain=c_sw,
        )

    if os.path.isfile(args.baseline_csv):
        base_full = pm.label_codes(pm.read_label_csv(args.baseline_csv))
    else:
        base_full = pred_full.copy()

    m_new = _compute_metrics_bundle(gt_full, pred_full, lp, m, pipe.decode_cfg, sw_sig, w_start, w_end)
    m_base = _compute_metrics_bundle(gt_full, base_full, None, m, pipe.decode_cfg, None, w_start, w_end)

    gt_w = gt_full[w_start:w_end]
    pred_w = pred_full[w_start:w_end].copy()
    pred_w[gt_w == X] = X
    base_w = base_full[w_start:w_end].copy()
    base_w[gt_w == X] = X
    vit_w = vit_raw[w_start:w_end].copy()
    vit_w[gt_w == X] = X
    vis_w = np.isfinite(m.ball[w_start:w_end, 0])

    err_codes, spell_rows, tr_rows = _build_inspectors(gt_w, pred_w, vis_w, w_start)

    # Compact quantized frame arrays for fast client playback
    def _q255(arr: Optional[np.ndarray]) -> List[int]:
        if arr is None:
            return []
        return np.round(np.clip(arr[w_start:w_end], 0.0, 1.0) * 255.0).astype(np.uint8).tolist()

    p2 = np.exp(lp[w_start:w_end].astype(np.float64))
    p_d2 = np.clip(p2[:, D], 0.0, 1.0)
    p_h2 = p2[:, H] / np.maximum(p2[:, H] + p2[:, A], 1e-6)

    pos_w = m.pos[w_start:w_end]
    players_q = np.where(np.isfinite(pos_w), np.round(pos_w * 10.0), -32768).astype(np.int16)
    rb_xy_q = np.round(np.nan_to_num(rb.ball_hat[w_start:w_end], nan=-3276.8) * 10.0).astype(np.int16)
    raw_ball_w = m.ball[w_start:w_end]
    raw_ball_q = np.where(np.isfinite(raw_ball_w), np.round(raw_ball_w * 10.0), -32768).astype(np.int16)

    payload = {
        "model_tag": args.model_tag,
        "fps": FPS,
        "w_start": w_start,
        "w_end": w_end,
        "n_win": int(w_end - w_start),
        "pitch": [float(m.pitch[0]), float(m.pitch[1])],
        "n_slots": int(m.pos.shape[1]),
        "team": [int(t) for t in m.team],
        "metrics_new": m_new,
        "metrics_base": m_base,
        "spells": spell_rows,
        "transitions": tr_rows,
        "gt_w": gt_w.tolist(),
        "pred_w": pred_w.tolist(),
        "base_w": base_w.tolist(),
        "vit_raw_w": vit_w.tolist(),
        "err_codes": err_codes,
        "rb_gap": rb.gap_len[w_start:w_end].astype(np.int16).tolist(),
        "p_bconf": _q255(rb.ball_conf),
        "p_d2": np.round(np.clip(p_d2, 0.0, 1.0) * 255.0).astype(np.uint8).tolist(),
        "p_h2": np.round(np.clip(p_h2, 0.0, 1.0) * 255.0).astype(np.uint8).tolist(),
        "p_ha1": _q255(pha1),
        "p_ha_bf": _q255(pha_bf),
        "p_ha_nn": _q255(pha_nn),
        "p_sw": _q255(sw_sig),
        "players_q": players_q.reshape(w_end - w_start, -1).tolist(),
        "rb_xy_q": rb_xy_q.reshape(-1).tolist(),
        "raw_ball_q": raw_ball_q.reshape(-1).tolist(),
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    html_text = _render_html(payload_json)

    for out_p in [args.out_html, args.copy_report]:
        if not out_p:
            continue
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_text)
        size_mb = os.path.getsize(out_p) / (1024.0 * 1024.0)
        print(f"Wrote live 5-min visualizer to {out_p} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()

