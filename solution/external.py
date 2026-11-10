"""External tuning games: Metrica Sports public sample data (Games 1-3).

Source: https://github.com/metrica-sports/sample-data  (anonymised, corrected tracking + events).
These are OTHER games, so the rules allow tuning and training on them. Nothing here touches the
competition match.

Two jobs:
  1. load()      clean tracking -> Match, plus a proxy H/A/D ground truth derived from the data.
  2. degrade()   make the clean tracking look like automatic tracking (ball gaps, noise, dropouts),
                 so parameters tuned here have a chance of transferring.

Proxy ground truth (our derivation, see docs/DATA_CONTRACT.md):
  D      where the corrected tracking has no ball (it is NaN exactly while the ball is out of play)
  H / A  otherwise, the team of the most recent on-ball event (pass, carry, shot, recovery,
         set piece). Challenges are ignored: a touch is not possession.
  A stint of the other team shorter than UNCONTROLLED_MAX_FRAMES with no completed pass, carry or
  shot, sandwiched inside one team's possession, is treated as an uncontrolled touch and merged.
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .common import A, CACHE_DIR, D, EXTERNAL_DIR, FPS, H, X, Match

METRICA_DIR = os.path.join(EXTERNAL_DIR, "metrica")
GAMES = ("metrica1", "metrica2", "metrica3")
PITCH = (105.0, 68.0)
PAD = 250                      # ungraded frames inserted between the halves
OWNER_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}
CONTROL_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE"}
UNCONTROLLED_MAX_FRAMES = 50   # 2 s
MAX_SPEED = 12.0


def _speeds(pos):
    """(n, S) speed in m/s from positions, central difference over +/-2 frames."""
    spd = np.full(pos.shape[:2], np.nan, dtype=np.float32)
    d = pos[4:] - pos[:-4]
    spd[2:-2] = np.hypot(d[..., 0], d[..., 1]) * (FPS / 4.0)
    return np.minimum(spd, MAX_SPEED)


def derive_gt(n, blocks, ball_missing, events, merge=True):
    """Proxy H/A/D labels. events: [(start_idx, team, type)] sorted by start_idx.

    merge=False keeps every change of event team as a change of possession (no uncontrolled-touch
    rule). The benchmark scores against both variants, because the rule is our reading of the
    definition and a model tuned to one reading only would be tuned to our own assumption.
    """
    gt = np.full(n, X, dtype=np.int8)
    owner = np.full(n, -1, dtype=np.int8)
    control = np.zeros(n, dtype=bool)          # frame where a controlled action starts
    for start, team, kind in events:
        if kind in OWNER_TYPES and 0 <= start < n:
            owner[start] = team
            if kind in CONTROL_TYPES:
                control[start] = True
    for b0, b1 in blocks:
        # forward-fill the owner inside the block
        seg = owner[b0:b1]
        idx = np.where(seg >= 0, np.arange(len(seg)), -1)
        np.maximum.accumulate(idx, out=idx)
        filled = np.where(idx >= 0, seg[np.maximum(idx, 0)], -1)
        dead = ball_missing[b0:b1] | (filled < 0)
        lab = np.where(dead, D, filled).astype(np.int8)
        if merge:
            lab = _merge_uncontrolled(lab, control[b0:b1])
        gt[b0:b1] = lab
    return gt


def _runs(lab):
    breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(lab)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(lab[bounds[i]])) for i in range(len(bounds) - 1)]


def _merge_uncontrolled(lab, control):
    lab = lab.copy()
    changed = True
    while changed:
        changed = False
        runs = _runs(lab)
        for i in range(1, len(runs) - 1):
            s, e, k = runs[i]
            if k == D or e - s >= UNCONTROLLED_MAX_FRAMES:
                continue
            prev_k, next_k = runs[i - 1][2], runs[i + 1][2]
            if prev_k == next_k and prev_k != D and prev_k != k and not control[s:e].any():
                lab[s:e] = prev_k
                changed = True
                break
    return lab


def _assemble(name, periods, home, away, ball, events_raw):
    """periods: per-row period id (1/2). home/away: (rows, S, 2) normalised. ball: (rows, 2).

    events_raw: [(frame_row_index, team, type)] with row indices into the unpadded arrays.
    """
    rows = len(periods)
    split = int(np.searchsorted(periods, 2))
    n = rows + PAD
    index = np.arange(rows) + np.where(np.arange(rows) >= split, PAD, 0)

    def padded(arr):
        out = np.full((n,) + arr.shape[1:], np.nan, dtype=np.float32)
        out[index] = arr
        return out

    scale = np.array(PITCH, dtype=np.float32)
    pos = padded(np.concatenate([home, away], axis=1)) * scale
    ball_m = padded(ball) * scale
    team = np.concatenate([np.zeros(home.shape[1], np.int8), np.ones(away.shape[1], np.int8)])
    blocks = [(0, split), (split + PAD, n)]
    events = sorted((int(index[min(max(r, 0), rows - 1)]), t, k) for r, t, k in events_raw)
    missing = np.isnan(ball_m[:, 0])
    gt = derive_gt(n, blocks, missing, events)
    gt_raw = derive_gt(n, blocks, missing, events, merge=False)
    return Match(name=name, ball=ball_m, pos=pos, spd=_speeds(pos), team=team, blocks=blocks,
                 pitch=PITCH, gt=gt, meta={"source": "metrica sample-data", "clean": True,
                                           "gt_raw": gt_raw})


def _load_csv_game(g):
    def read(side):
        path = os.path.join(METRICA_DIR, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (g, side))
        return pd.read_csv(path, skiprows=2)
    home, away = read("Home"), read("Away")
    periods = home["Period"].values
    first_frame = int(home["Frame"].iloc[0])
    hv = home.iloc[:, 3:].values.astype(np.float32)
    av = away.iloc[:, 3:].values.astype(np.float32)
    ball = hv[:, -2:]
    hp = hv[:, :-2].reshape(len(hv), -1, 2)
    ap = av[:, :-2].reshape(len(av), -1, 2)
    ev = pd.read_csv(os.path.join(METRICA_DIR, "Sample_Game_%d_RawEventsData.csv" % g))
    events = [(int(f) - first_frame, 0 if t == "Home" else 1, str(k))
              for f, t, k in zip(ev["Start Frame"], ev["Team"], ev["Type"])]
    return _assemble("metrica%d" % g, periods, hp, ap, ball, events)


def _load_epts_game():
    meta = ET.parse(os.path.join(METRICA_DIR, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in meta.iter("Team")]
    player_team = {p.get("id"): team_ids.index(p.get("teamId")) for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}
    second_start = int(params["second_half_start"])

    # Each DataFormatSpecification lists the channels for a frame range (they change at subs).
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        teams = [player_team[channel_player[c]] for c in refs[0::2]]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), teams))

    frames, rows, balls = [], [], []
    with open(os.path.join(METRICA_DIR, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            fid, players, ball = line.strip().split(":")
            frames.append(int(fid))
            rows.append(players.replace(";", ","))
            balls.append(ball)
    frames = np.array(frames)
    assert np.array_equal(frames, np.arange(1, len(frames) + 1))
    flat = np.array([np.array(r.split(","), dtype=np.float32) for r in rows])
    ball = np.array([np.array(b.split(","), dtype=np.float32) for b in balls])
    slots = flat.reshape(len(flat), -1, 2)

    # Route every slot to a home or an away column block; a slot's team is fixed within a spec.
    S = slots.shape[1]
    home = np.full((len(slots), S, 2), np.nan, dtype=np.float32)
    away = np.full((len(slots), S, 2), np.nan, dtype=np.float32)
    for start, end, teams in specs:
        sl = slice(start - 1, end)
        for j, t in enumerate(teams):
            (home if t == 0 else away)[sl, j] = slots[sl, j]
    keep_h = ~np.isnan(home[..., 0]).all(axis=0)
    keep_a = ~np.isnan(away[..., 0]).all(axis=0)
    periods = np.where(frames >= second_start, 2, 1)

    ev = json.load(open(os.path.join(METRICA_DIR, "Sample_Game_3_events.json")))["data"]
    events = [(int(e["start"]["frame"]) - 1, team_ids.index(e["team"]["id"]), e["type"]["name"])
              for e in ev if e.get("team") and e["start"].get("frame") is not None]
    return _assemble("metrica3", periods, home[:, keep_h], away[:, keep_a], ball, events)


def load(name):
    """Clean Match with proxy ground truth, cached under cache/."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "ext3_%s.npz" % name)
    if os.path.isfile(path):
        z = np.load(path)
        return Match(name=name, ball=z["ball"], pos=z["pos"], spd=z["spd"], team=z["team"],
                     blocks=[tuple(b) for b in z["blocks"].tolist()], pitch=PITCH, gt=z["gt"],
                     meta={"source": "metrica sample-data", "clean": True, "gt_raw": z["gt_raw"]})
    m = _load_epts_game() if name == "metrica3" else _load_csv_game(int(name[-1]))
    np.savez_compressed(path, ball=m.ball, pos=m.pos, spd=m.spd, team=m.team,
                        blocks=np.array(m.blocks), gt=m.gt, gt_raw=m.meta["gt_raw"])
    return m


# -- degradation ---------------------------------------------------------------------------------
# Published facts about the competition tracking that the degradation aims at (DATA.md):
#   ball missing on 41% of graded frames; ~20.5 tracked objects per graded frame out of 22.
#
# WHAT THE BALL DOES DURING A STOPPAGE IS UNKNOWN TO US. The clean data has no ball there, so
# anything we put in is invented. Benchmark v1/v2 invented one picture and then tuned dead-ball
# cues (ball outside the pitch, ball stationary) against it, which is circular. From v3 on:
#   * no model may use the ball to decide dead vs alive (see model.py / learned.py), and
#   * the benchmark runs three different invented pictures, so any remaining dependence on the
#     invention shows up as a spread between them instead of hiding inside one number.
SCENARIOS = {
    #            ball-missing rate in play / in stoppages, mean gap, what the ball does when dead
    "path":   dict(p_alive=0.41, p_dead=0.41, gap=30, dead_ball="path"),    # fetched and placed
    "hold":   dict(p_alive=0.35, p_dead=0.50, gap=30, dead_ball="hold"),    # stays where it stopped
    "hidden": dict(p_alive=0.30, p_dead=1.00, gap=40, dead_ball="hidden"),  # never tracked when dead
}
SIGMA_BALL, PLAYER_DROP = 0.25, 0.07


def _burst_mask(n, rate, mean_len, rng):
    """Boolean mask, True in bursts of geometric length, with stationary share `rate`."""
    if rate <= 0:
        return np.zeros(n, dtype=bool)
    if rate >= 1:
        return np.ones(n, dtype=bool)
    p_end = 1.0 / mean_len
    p_start = p_end * rate / (1.0 - rate)
    u = rng.random(n)
    mask = np.zeros(n, dtype=bool)
    state = rng.random() < rate
    for i in range(n):
        if state:
            if u[i] < p_end:
                state = False
        elif u[i] < p_start:
            state = True
        mask[i] = state
    return mask


def _fill_dead_ball(ball, blocks, pitch, rng, mode="path"):
    """Give the ball a position during stoppages (the clean data has none).

    Assumed picture of what an automatic tracker sees, when it sees anything:
      * the ball left the pitch near a line: it runs 1-8 m past that line, lies there, is fetched
        back to the restart spot, then sits still until play restarts;
      * the stoppage started in the field (a foul): it moves slowly to the restart spot and sits.
    These are assumptions from how football works, not statistics measured on any tracking file.
    """
    L, W = pitch
    out = ball.copy()
    for b0, b1 in blocks:
        seg = out[b0:b1]
        missing = np.isnan(seg[:, 0])
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            before = seg[s - 1].copy() if s > 0 else None
            after = seg[e].copy() if e < len(seg) else None
            if before is None and after is None:
                continue
            p0 = before if before is not None else after
            p1 = after if after is not None else before
            u = np.arange(e - s) / float(max(1, e - s))
            gaps = np.array([p0[0], L - p0[0], p0[1], W - p0[1]])
            side = int(np.argmin(gaps))
            if mode == "hold":
                knots_u, knots = [0.0, 1.0], [p0, p0]
            elif gaps[side] < 3.0:
                q = p0.copy()
                push = rng.uniform(1.0, 8.0)
                q[side // 2] = (-push, L + push, -push, W + push)[side]
                knots_u = [0.0, 0.15, 0.45, 0.70, 1.0]
                knots = [p0, q, q, p1, p1]
            else:
                knots_u = [0.0, 0.60, 1.0]
                knots = [p0, p1, p1]
            for k in (0, 1):
                seg[s:e, k] = np.interp(u, knots_u, [p[k] for p in knots])
    return out


def degrade(match, scenario="path", seed=0):
    """A copy of a clean external Match that looks like automatic tracking."""
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    n = match.n
    dead = match.gt == D
    ball = _fill_dead_ball(match.ball, match.blocks, match.pitch, rng, cfg["dead_ball"])
    ball = ball + rng.normal(0, SIGMA_BALL, ball.shape).astype(np.float32)
    lost = np.where(dead, _burst_mask(n, cfg["p_dead"], cfg["gap"], rng),
                    _burst_mask(n, cfg["p_alive"], cfg["gap"], rng))
    ball[lost] = np.nan

    pos = match.pos + rng.normal(0, 0.10, match.pos.shape).astype(np.float32)
    spd = match.spd + rng.normal(0, 0.15, match.spd.shape).astype(np.float32)
    for j in range(pos.shape[1]):
        gone = _burst_mask(n, PLAYER_DROP, 50, rng)
        pos[gone, j] = np.nan
        spd[gone, j] = np.nan
    spd = np.clip(spd, 0, MAX_SPEED)
    return Match(name="%s/%s" % (match.name, scenario), ball=ball.astype(np.float32), pos=pos,
                 spd=spd, team=match.team, blocks=match.blocks, pitch=match.pitch, gt=match.gt,
                 meta={"source": match.meta.get("source"), "scenario": scenario, "seed": seed,
                       "gt_raw": match.meta["gt_raw"]})


if __name__ == "__main__":
    import possession_metrics as pm
    for name in GAMES:
        m = load(name)
        runs = pm.valid_runs(m.gt)
        stats = pm.possession_stats(m.gt, runs)
        graded = m.gt != X
        minutes = graded.sum() / FPS / 60.0
        raw = pm.possession_stats(m.meta["gt_raw"], runs)
        print("%s  frames %d  graded %.1f min  D %.1f%%  possessions %d (%.2f/min)  mean %.1f s  "
              "home share %.2f   | without the uncontrolled-touch rule: %d possessions, mean %.1f s"
              % (name, m.n, minutes, 100 * np.mean(m.gt[graded] == D), stats["n_spells"],
                 stats["n_spells"] / minutes, stats["mean_duration_s"], stats["home_share"],
                 raw["n_spells"], raw["mean_duration_s"]))
        for sc in SCENARIOS:
            d = degrade(m, sc)
            vis = ~np.isnan(d.ball[:, 0])
            print("   %-7s ball missing %.1f%% of graded, tracked objects %.1f" % (
                sc, 100 * np.mean(~vis[graded]),
                np.mean((~np.isnan(d.pos[graded][:, :, 0])).sum(axis=1))))
