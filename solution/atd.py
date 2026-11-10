"""Load the competition match (BAR-ATH tactical ATD + Smart Tagging) as a Match.

DATA CONTRACT: this module reads gated data. It prints aggregates only -- never rows, positions
or per-frame values -- and its cache (cache/) is gitignored.
"""
import os

import numpy as np

from .common import CACHE_DIR, Match

import games  # noqa: E402
import baseline as starter_baseline  # noqa: E402  (starter/baseline.py)
import load_smart_tagging  # noqa: E402

KICKOFF_2ND = 113742      # published in FAQ.md: graded from 110808, all D until this frame


def _parse(raw_path, n_frames, n_slots):
    pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    spd = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    bad = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                bad += 1
                continue
            t = int(parts[0]) - 1          # frame_id counts from 1
            if not 0 <= t < n_frames:
                bad += 1
                continue
            for j, group in enumerate(parts[1].split(";")):
                if j >= n_slots or not group or group[0] == "N":
                    continue
                f = group.split(",")
                pos[t, j, 0], pos[t, j, 1] = float(f[0]), float(f[1])
                if len(f) > 2:
                    spd[t, j] = float(f[2])
            b = parts[2]
            if b and b[0] != "N":
                f = b.split(",")
                ball[t, 0], ball[t, 1] = float(f[0]), float(f[1])
    return pos, spd, ball, bad


def load(game=games.GAME):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "%s_tactical.npz" % game)
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game)
    team = starter_baseline.team_of_each_track(meta_path)
    n = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    if os.path.isfile(path):
        z = np.load(path)
        pos, spd, ball = z["pos"], z["spd"], z["ball"]
    else:
        pos, spd, ball, bad = _parse(raw_path, n, len(team))
        if bad:
            print("warning: %d unparseable ATD lines" % bad)
        scale = np.array(pitch, dtype=np.float32)
        pos, ball = pos * scale, ball * scale
        np.savez_compressed(path, pos=pos, spd=spd, ball=ball)
    blocks = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    tags = load_smart_tagging.load(game)
    return Match(name=game, ball=ball, pos=pos, spd=spd, team=team.astype(np.int8), blocks=blocks,
                 pitch=pitch, gt=None, tags=tags, meta={"source": "competition ATD"})


def summary(m):
    """Aggregate description, limited to figures the organisers already publish (DATA.md), so it
    works as a loader check and tells us nothing new about the match. An earlier version also
    printed two unpublished statistics; they were removed (audit finding, RESEARCH_LOG E006)."""
    graded = np.zeros(m.n, dtype=bool)
    for b0, b1 in m.blocks:
        graded[b0:b1] = True
    vis = ~np.isnan(m.ball[:, 0])
    tracked = (~np.isnan(m.pos[:, :, 0])).sum(axis=1)
    return {
        "frames": m.n, "graded": int(graded.sum()), "slots": int(len(m.team)),
        "ball_missing_graded_pct": round(100 * float(np.mean(~vis[graded])), 1),
        "tracked_objects_graded_mean": round(float(tracked[graded].mean()), 1),
        "n_tags": len(m.tags or []),
    }


if __name__ == "__main__":
    for k, v in summary(load()).items():
        print("%-36s %s" % (k, v))
