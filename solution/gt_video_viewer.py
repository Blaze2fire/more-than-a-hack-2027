"""Local video + public ground-truth viewer, for understanding what counts as possession.

    .venv/bin/python -m solution.gt_video_viewer [--video PATH] [--out PATH]

Writes a self-contained HTML page that plays the tactical video and shows the public GT label of
the current frame, a clickable spell timeline and a spell list. Open it in a local browser.

Rules (RULES.md, organizer email): inspecting the provided data and video to understand possession
is allowed; checking our output against the video is not. So this tool takes no prediction input
by design -- do not add one. It prints only the output path; the HTML embeds GT labels, so it stays
local (solution/out/ is git-ignored) and must not be shared with or opened by an AI assistant.
"""
import argparse
import json
import os
import sys

import numpy as np

from .common import FPS, ROOT

if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_VIDEOS = (
    os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4"),
    os.path.join(ROOT, "data", "BAR-ATH_tactical-002.mp4"),
)
DEFAULT_OUT = os.path.join(ROOT, "solution", "out", "gt_video_viewer.html")


def runs(labels):
    """Run-length encode a label-string array into [start, end_inclusive, label] lists."""
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(labels) - 1]])
    return [[int(s), int(e), str(labels[s])] for s, e in zip(starts, ends)]


def build_html(segments, n_frames, video_src, fps=FPS):
    data = json.dumps({"segments": segments, "n": int(n_frames), "fps": fps, "video": video_src})
    return PAGE.replace("/*__DATA__*/null", data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", default=next((v for v in DEFAULT_VIDEOS if os.path.exists(v)),
                                            DEFAULT_VIDEOS[0]))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    labels = pm.read_label_csv(PUBLIC_GT)["label"].astype(str).str.strip().str.upper().values
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    video_src = os.path.relpath(os.path.abspath(args.video), os.path.dirname(out))
    with open(out, "w") as f:
        f.write(build_html(runs(labels), len(labels), video_src))
    print("wrote %s (open locally in a browser)" % out)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BAR-ATH GT Viewer</title>
<style>
:root { --bg:#111418; --panel:#1b2027; --fg:#e8eaed; --mut:#9aa3ad; --line:#2c333c;
        --H:#2f6fde; --A:#d8443a; --D:#8a8f96; --X:#2a2f36; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
.wrap { display:grid; grid-template-columns: minmax(0,1fr) 320px; gap:12px; padding:12px; height:100vh; }
.main { display:flex; flex-direction:column; gap:10px; min-width:0; }
video { width:100%; max-height:68vh; background:#000; border-radius:6px; }
.bar { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
.badge { font-size:22px; font-weight:700; padding:6px 16px; border-radius:6px; min-width:260px; text-align:center; }
.mono { font-family: ui-monospace, monospace; }
.mut { color:var(--mut); }
canvas { width:100%; height:56px; border-radius:4px; cursor:pointer; background:var(--X); display:block; }
.side { background:var(--panel); border-radius:6px; overflow:auto; }
.side h2 { font-size:13px; margin:0; padding:10px 12px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--panel); }
.seg { display:flex; gap:8px; padding:5px 12px; cursor:pointer; border-left:4px solid transparent; }
.seg:hover { background:#232a33; }
.seg.cur { background:#2a323d; }
button, select, label { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:4px 8px; font:inherit; }
.keys { font-size:12px; }
.err { color:#ff8a80; }
@media (max-width: 800px) { .wrap { grid-template-columns: 1fr; height:auto; } .side { max-height:50vh; } }
</style></head><body>
<div class="wrap">
  <div class="main">
    <video id="v" controls preload="auto"></video>
    <div id="err" class="err"></div>
    <div class="bar">
      <div id="badge" class="badge">-</div>
      <div class="mono">frame <b id="fr">0</b> &middot; <span id="tm">0:00.00</span></div>
      <div class="mono mut" id="spell"></div>
    </div>
    <canvas id="tl"></canvas>
    <div class="bar">
      <button id="prev">&#9664; prev change</button>
      <button id="next">next change &#9654;</button>
      <select id="rate"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
      <label><input type="checkbox" id="full"> full match timeline</label>
      <span class="keys mut">space play/pause &middot; &larr;/&rarr; 1 frame &middot; shift+&larr;/&rarr; 1 s &middot; [ / ] prev/next change</span>
    </div>
  </div>
  <div class="side"><h2>Labelled spells (public GT)</h2><div id="list"></div></div>
</div>
<script>
const DATA = /*__DATA__*/null;
const FPS = DATA.fps, SEGS = DATA.segments, N = DATA.n;
const NAME = {H:"H · Barcelona", A:"A · Athletic", D:"D · dead ball", X:"X · not labelled"};
const css = k => getComputedStyle(document.documentElement).getPropertyValue("--" + k).trim();
const v = document.getElementById("v"), tl = document.getElementById("tl");
v.src = DATA.video;
v.onerror = () => document.getElementById("err").textContent =
  "Video failed to load from " + DATA.video + " (wrong path or a codec this browser can't play; try Chrome, or re-run with --video).";

const labelled = SEGS.filter(s => s[2] !== "X");
const lo0 = labelled.length ? labelled[0][0] : 0, hi0 = labelled.length ? labelled[labelled.length - 1][1] : N - 1;
const pad = Math.round(10 * FPS);
let view = [Math.max(0, lo0 - pad), Math.min(N - 1, hi0 + pad)];

function segAt(f) {  // binary search
  let a = 0, b = SEGS.length - 1;
  while (a < b) { const m = (a + b + 1) >> 1; if (SEGS[m][0] <= f) a = m; else b = m - 1; }
  return a;
}
const frameOf = t => Math.min(N - 1, Math.max(0, Math.floor(t * FPS + 1e-3)));
const seek = f => { v.currentTime = (Math.max(0, Math.min(N - 1, f)) + 0.5) / FPS; };
const fmt = f => { const s = f / FPS, m = Math.floor(s / 60); return m + ":" + (s - 60 * m).toFixed(2).padStart(5, "0"); };

// spell list
const list = document.getElementById("list");
const rows = labelled.map(s => {
  const d = document.createElement("div");
  d.className = "seg mono"; d.style.borderLeftColor = css(s[2]);
  d.innerHTML = `<b style="width:14px">${s[2]}</b><span>${s[0]}–${s[1]}</span><span class="mut">${((s[1] - s[0] + 1) / FPS).toFixed(2)} s</span>`;
  d.onclick = () => seek(s[0]);
  list.appendChild(d); return d;
});

function draw(f) {
  const dpr = window.devicePixelRatio || 1, W = tl.clientWidth, Hh = tl.clientHeight;
  tl.width = W * dpr; tl.height = Hh * dpr;
  const g = tl.getContext("2d"); g.scale(dpr, dpr);
  const [a, b] = view, sx = W / (b - a + 1);
  for (let i = segAt(a); i < SEGS.length && SEGS[i][0] <= b; i++) {
    const s = SEGS[i], x0 = (Math.max(s[0], a) - a) * sx, x1 = (Math.min(s[1], b) - a + 1) * sx;
    g.fillStyle = css(s[2]); g.fillRect(x0, 0, Math.max(1, x1 - x0), Hh);
  }
  if (f >= a && f <= b) { g.fillStyle = "#fff"; g.fillRect((f - a) * sx - 1, 0, 2, Hh); }
}

let lastSeg = -1;
function update(f) {
  const i = segAt(f), s = SEGS[i];
  document.getElementById("fr").textContent = f;
  document.getElementById("tm").textContent = fmt(f);
  const b = document.getElementById("badge");
  b.textContent = NAME[s[2]] || s[2]; b.style.background = css(s[2]);
  document.getElementById("spell").textContent =
    `spell ${s[0]}–${s[1]} · ${((s[1] - s[0] + 1) / FPS).toFixed(2)} s · +${((f - s[0]) / FPS).toFixed(2)} s in`;
  if (i !== lastSeg) {
    rows.forEach(r => r.classList.remove("cur"));
    const k = labelled.indexOf(s);
    if (k >= 0) { rows[k].classList.add("cur"); rows[k].scrollIntoView({block: "nearest"}); }
    lastSeg = i;
  }
  draw(f);
}

// frame-accurate updates where supported, else timeupdate + rAF while playing
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  const cb = (_, md) => { update(frameOf(md.mediaTime)); v.requestVideoFrameCallback(cb); };
  v.requestVideoFrameCallback(cb);
} else {
  const loop = () => { update(frameOf(v.currentTime)); if (!v.paused) requestAnimationFrame(loop); };
  v.addEventListener("play", loop);
}
v.addEventListener("seeked", () => update(frameOf(v.currentTime)));
v.addEventListener("loadedmetadata", () => { if (v.currentTime === 0) seek(lo0); });

tl.addEventListener("click", e => {
  const r = tl.getBoundingClientRect();
  seek(Math.round(view[0] + (e.clientX - r.left) / r.width * (view[1] - view[0] + 1)));
});
document.getElementById("full").onchange = e => {
  view = e.target.checked ? [0, N - 1] : [Math.max(0, lo0 - pad), Math.min(N - 1, hi0 + pad)];
  draw(frameOf(v.currentTime));
};
document.getElementById("rate").onchange = e => { v.playbackRate = +e.target.value; };
const jump = dir => {
  const i = segAt(frameOf(v.currentTime)), j = Math.max(0, Math.min(SEGS.length - 1, i + dir));
  seek(dir < 0 && frameOf(v.currentTime) > SEGS[i][0] ? SEGS[i][0] : SEGS[j][0]);
};
document.getElementById("prev").onclick = () => jump(-1);
document.getElementById("next").onclick = () => jump(1);
window.addEventListener("resize", () => draw(frameOf(v.currentTime)));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  const f = frameOf(v.currentTime), step = e.shiftKey ? FPS : 1;
  if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); v.pause(); seek(f + step); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); v.pause(); seek(f - step); }
  else if (e.key === "]") jump(1);
  else if (e.key === "[") jump(-1);
});
update(0);
</script></body></html>
"""

if __name__ == "__main__":
    main()
