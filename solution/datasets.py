"""Load, normalize, cache, and realistically degrade external tracking datasets.

Supported datasets (all rule-compliant external football tracking datasets):
- Sportec DFL v1 (CC BY 4.0): 7 matches @ 25 Hz with BallStatus, BallPossession,
  and observed ball trajectories even during stoppages; plus ELASTIC (CC BY 4.0)
  human-checked pass/reception/out-of-bounds event timestamps for 3 matches.
- WCSF Groningen (MIT): 1 match @ 10 Hz (de-mirrored & resampled to 25 Hz, with
  inactive precedence and conflict masking).
- SkillCorner Open Data current (MIT): 20 matches @ 10 Hz (resampled to 25 Hz;
  supervised only for live team H/A possession & ball imputation; dead frames
  are marked X so camera stoppages never leak visibility).
- Metrica Sample Data (1, 2, 3): delegated to solution.external when needed.

All datasets normalize pitch coordinates to [0, pitch_L] x [0, pitch_W] metres and
compute player speeds identically from lightly smoothed positions so kinematics
match BAR-ATH tactical ATD without unit or differentiation mismatch.
"""
import glob
import json
import os
import zipfile
from typing import Dict, List, Optional, Tuple

from lxml import etree
import numpy as np
import pandas as pd

from solution.common import A, CACHE_DIR, D, EXTERNAL_DIR, FPS, H, ROOT, X, Match

DFL_DIR = os.path.join(EXTERNAL_DIR, "sportec-dfl-v1")
ELASTIC_DIR = os.path.join(EXTERNAL_DIR, "elastic-annotations", "benchmark", "gt")
GRON_DIR = os.path.join(EXTERNAL_DIR, "wcsf-groningen")
SC_CUR_DIR = os.path.join(EXTERNAL_DIR, "skillcorner-current", "data", "matches")

DFL_GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
ELASTIC_GAMES = ("J03WMX", "J03WN1", "J03WPY")

MAX_PLAYER_SPEED = 12.0
MAX_BALL_JUMP_SPEED = 38.0


def compute_speed(pos: np.ndarray, blocks: List[Tuple[int, int]], max_spd: float = MAX_PLAYER_SPEED) -> np.ndarray:
    """Compute unified m/s player speeds from positions across each contiguous block.

    Lightly smooths valid 3-frame neighborhoods to suppress high-frequency camera
    jitter before central-difference differentiation (±2 frames = 0.16s), falling
    back to 1-frame edges so tracklet starts/ends still retain finite speed estimates.
    """
    n, s, _ = pos.shape
    spd = np.full((n, s), np.nan, dtype=np.float32)
    for b0, b1 in blocks:
        if b1 - b0 < 3:
            continue
        p = pos[b0:b1].astype(np.float64, copy=False)
        valid = np.isfinite(p[:, :, 0]) & np.isfinite(p[:, :, 1])
        # Smooth with 3-point kernel [0.25, 0.5, 0.25] only where neighbors exist.
        sm = p.copy()
        mid_ok = valid[1:-1] & valid[:-2] & valid[2:]
        sm[1:-1] = np.where(mid_ok[:, :, None], 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:], p[1:-1])

        out = np.full((b1 - b0, s), np.nan, dtype=np.float64)
        # Central difference across +/-2 frames (dt = 0.16s at 25 Hz)
        c2 = valid[:-4] & valid[4:]
        d2 = np.hypot(sm[4:, :, 0] - sm[:-4, :, 0], sm[4:, :, 1] - sm[:-4, :, 1]) * (FPS / 4.0)
        out[2:-2] = np.where(c2, d2, out[2:-2])

        # Fallback central +/-1 frame (dt = 0.08s) where +/-2 unavailable
        c1 = np.isnan(out[1:-1]) & valid[:-2] & valid[2:]
        d1 = np.hypot(sm[2:, :, 0] - sm[:-2, :, 0], sm[2:, :, 1] - sm[:-2, :, 1]) * (FPS / 2.0)
        out[1:-1] = np.where(c1, d1, out[1:-1])

        # Fallback one-sided +/-2 or +/-1 at edges
        f1 = np.isnan(out[:-1]) & valid[:-1] & valid[1:]
        df = np.hypot(sm[1:, :, 0] - sm[:-1, :, 0], sm[1:, :, 1] - sm[:-1, :, 1]) * FPS
        out[:-1] = np.where(f1, df, out[:-1])

        b1m = np.isnan(out[1:]) & valid[:-1] & valid[1:]
        out[1:] = np.where(b1m, df, out[1:])

        out[out > max_spd * 1.4] = np.nan
        spd[b0:b1] = np.clip(out, 0.0, max_spd).astype(np.float32)
    return spd


def standardize_match(m: Match) -> Match:
    """Recompute speeds identically from positions on any Match instance."""
    m.spd = compute_speed(m.pos, m.blocks)
    return m


def _cache_save(path: str, m: Match) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array([m.name]),
        ball=m.ball.astype(np.float32),
        pos=m.pos.astype(np.float32),
        spd=m.spd.astype(np.float32),
        team=m.team.astype(np.int8),
        blocks=np.array(m.blocks, dtype=np.int32),
        pitch=np.array(m.pitch, dtype=np.float32),
        gt=m.gt.astype(np.int8) if m.gt is not None else np.array([], dtype=np.int8),
        meta_json=np.array([json.dumps(m.meta)]),
    )


def _cache_load(path: str) -> Match:
    z = np.load(path, allow_pickle=False)
    gt_arr = z["gt"]
    return Match(
        name=str(z["name"][0]),
        ball=z["ball"].astype(np.float32),
        pos=z["pos"].astype(np.float32),
        spd=z["spd"].astype(np.float32),
        team=z["team"].astype(np.int8),
        blocks=[(int(a), int(b)) for a, b in z["blocks"]],
        pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
        gt=gt_arr.astype(np.int8) if gt_arr.size > 0 else None,
        meta=json.loads(str(z["meta_json"][0])),
    )


def _load_elastic_events(mid: str, n1: int, h1_start: int, h2_start: int) -> Optional[dict]:
    path = os.path.join(ELASTIC_DIR, f"{mid}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    def _to_idx(period_series, frame_series):
        out = []
        for p, f in zip(period_series, frame_series):
            if pd.isna(f):
                continue
            fi = int(f)
            if int(p) == 1:
                out.append(fi - h1_start)
            elif int(p) == 2:
                out.append(n1 + (fi - h2_start))
        return out
    kicks = _to_idx(df["period_id"], df["frame_id"])
    recvs = _to_idx(df["period_id"], df["receive_frame_id"])
    types = df["spadl_type"].astype(str).tolist()
    return {"kick_idx": kicks, "recv_idx": recvs, "types": types}


def load_dfl(mid: str, force_rebuild: bool = False) -> Match:
    """Load one Sportec DFL v1 match (cached to cache/dfl_<mid>_v1.npz)."""
    cache_path = os.path.join(CACHE_DIR, f"dfl_{mid}_v1.npz")
    if not force_rebuild and os.path.exists(cache_path):
        return _cache_load(cache_path)

    info_matches = glob.glob(os.path.join(DFL_DIR, f"*DFL_02_01_matchinformation_*{mid}*.xml"))
    pos_matches = glob.glob(os.path.join(DFL_DIR, f"*DFL_04_03_positions_raw_observed_*{mid}*.xml"))
    if not info_matches or not pos_matches:
        raise FileNotFoundError(f"Missing DFL XML files for {mid} in {DFL_DIR}")

    info_tree = etree.parse(info_matches[0])
    gen = info_tree.find(".//General")
    env = info_tree.find(".//Environment")
    home_id = gen.get("HomeTeamId")
    away_id = gen.get("GuestTeamId")
    pitch_l = float(env.get("PitchX") or 105.0)
    pitch_w = float(env.get("PitchY") or 68.0)

    # Pass 1: discover half frame boundaries from BALL and unique player IDs per team.
    sec_bounds: Dict[str, Tuple[int, int]] = {}
    players_by_team: Dict[str, set] = {home_id: set(), away_id: set()}
    for _, el in etree.iterparse(pos_matches[0], events=("end",), tag="FrameSet"):
        tid = el.get("TeamId")
        gs = el.get("GameSection")
        if tid == "BALL":
            frames = el.findall("Frame")
            sec_bounds[gs] = (int(frames[0].get("N")), int(frames[-1].get("N")))
        elif tid in players_by_team:
            pid = el.get("PersonId")
            if pid:
                players_by_team[tid].add(pid)
        el.clear()

    h1_s, h1_e = sec_bounds["firstHalf"]
    h2_s, h2_e = sec_bounds["secondHalf"]
    n1 = h1_e - h1_s + 1
    n2 = h2_e - h2_s + 1
    n_tot = n1 + n2
    blocks = [(0, n1), (n1, n_tot)]

    home_pids = sorted(players_by_team[home_id])
    away_pids = sorted(players_by_team[away_id])
    pid_to_slot: Dict[str, int] = {}
    for i, pid in enumerate(home_pids):
        pid_to_slot[pid] = i
    offset = len(home_pids)
    for i, pid in enumerate(away_pids):
        pid_to_slot[pid] = offset + i
    n_slots = len(pid_to_slot)
    team = np.array([0] * len(home_pids) + [1] * len(away_pids), dtype=np.int8)

    ball = np.full((n_tot, 2), np.nan, dtype=np.float32)
    pos = np.full((n_tot, n_slots, 2), np.nan, dtype=np.float32)
    gt = np.full(n_tot, X, dtype=np.int8)

    half_cfg = {"firstHalf": (h1_s, h1_e, 0), "secondHalf": (h2_s, h2_e, n1)}
    hl_x, hl_y = 0.5 * pitch_l, 0.5 * pitch_w

    # Pass 2: fill arrays.
    for _, el in etree.iterparse(pos_matches[0], events=("end",), tag="FrameSet"):
        tid = el.get("TeamId")
        gs = el.get("GameSection")
        if gs not in half_cfg:
            el.clear()
            continue
        f_start, f_end, base = half_cfg[gs]
        if tid == "BALL":
            for f in el.findall("Frame"):
                n = int(f.get("N"))
                if f_start <= n <= f_end:
                    idx = base + (n - f_start)
                    xv, yv = f.get("X"), f.get("Y")
                    if xv is not None and yv is not None:
                        ball[idx, 0] = float(xv) + hl_x
                        ball[idx, 1] = float(yv) + hl_y
                    bs = f.get("BallStatus")
                    bp = f.get("BallPossession")
                    if bs == "0":
                        gt[idx] = D
                    elif bs == "1":
                        gt[idx] = H if bp == "1" else (A if bp == "2" else X)
        elif tid in players_by_team:
            slot = pid_to_slot.get(el.get("PersonId"))
            if slot is not None:
                for f in el.findall("Frame"):
                    n = int(f.get("N"))
                    if f_start <= n <= f_end:
                        xv, yv = f.get("X"), f.get("Y")
                        if xv is not None and yv is not None:
                            idx = base + (n - f_start)
                            pos[idx, slot, 0] = float(xv) + hl_x
                            pos[idx, slot, 1] = float(yv) + hl_y
        el.clear()

    spd = compute_speed(pos, blocks)
    elastic = _load_elastic_events(mid, n1, h1_s, h2_s)
    meta = {
        "source": "sportec-dfl-v1",
        "match_id": mid,
        "home_id": home_id,
        "away_id": away_id,
        "has_real_dead_ball": True,
        "n_elastic_events": len(elastic["kick_idx"]) if elastic else 0,
    }
    if elastic:
        meta["elastic_kick_idx"] = elastic["kick_idx"]
        meta["elastic_recv_idx"] = elastic["recv_idx"]

    m = Match(
        name=f"dfl_{mid}",
        ball=ball,
        pos=pos,
        spd=spd,
        team=team,
        blocks=blocks,
        pitch=(pitch_l, pitch_w),
        gt=gt,
        meta=meta,
    )
    _cache_save(cache_path, m)
    return m


def load_all_dfl(force_rebuild: bool = False) -> List[Match]:
    return [load_dfl(mid, force_rebuild=force_rebuild) for mid in DFL_GAMES]


def _resample_10hz_to_25hz(
    ts_ms: np.ndarray,
    arr_10hz: np.ndarray,
    t25_ms: np.ndarray,
    max_gap_ms: float = 150.0,
) -> np.ndarray:
    """Linearly interpolate a (T10, ...) array at regular 40 ms grid across contiguous steps."""
    # ts_ms is strictly increasing in ms. Find right index in ts_ms for each t25.
    idx_hi = np.searchsorted(ts_ms, t25_ms, side="left")
    idx_hi = np.clip(idx_hi, 0, len(ts_ms) - 1)
    idx_lo = np.clip(idx_hi - 1, 0, len(ts_ms) - 1)
    same = idx_lo == idx_hi
    dt = (ts_ms[idx_hi] - ts_ms[idx_lo]).astype(np.float64)
    w = np.where(same | (dt <= 0), 0.0, (t25_ms - ts_ms[idx_lo]) / np.maximum(dt, 1.0))

    shape_tail = arr_10hz.shape[1:]
    bshape = (-1,) + (1,) * len(shape_tail)
    lo_vals = arr_10hz[idx_lo]
    hi_vals = arr_10hz[idx_hi]
    same_b = same.reshape(bshape)
    dt_ok_b = (dt <= max_gap_ms).reshape(bshape)
    valid_step = same_b | (dt_ok_b & np.isfinite(lo_vals) & np.isfinite(hi_vals))
    w_b = w.reshape(bshape)
    interp = np.where(same_b, lo_vals, (1.0 - w_b) * lo_vals + w_b * hi_vals)
    # Also allow exact match even if lo is NaN
    exact = (t25_ms == ts_ms[idx_hi]).reshape(bshape) & np.isfinite(hi_vals)
    interp = np.where(exact, hi_vals, interp)
    out = np.where(valid_step | exact, interp, np.nan).astype(np.float32)
    return out


def load_groningen(force_rebuild: bool = False) -> Match:
    """Load WCSF Groningen @ 10 Hz, de-mirror group_id==2, resample to 25 Hz."""
    cache_path = os.path.join(CACHE_DIR, "wcsf_groningen_25hz_v1.npz")
    if not force_rebuild and os.path.exists(cache_path):
        return _cache_load(cache_path)

    pitch_l, pitch_w = 105.0, 68.0
    hl_x, hl_y = 0.5 * pitch_l, 0.5 * pitch_w

    halves = []
    all_pids: Dict[int, set] = {1: set(), 2: set()}
    for fn in ("positions_FIRST_HALF.csv", "positions_SECOND_HALF.csv"):
        with zipfile.ZipFile(os.path.join(GRON_DIR, fn + ".zip")) as zf:
            with zf.open(fn) as f:
                df = pd.read_csv(f)
        m2 = df["group_id"] == 2
        df.loc[m2, ["x", "y", "x_ball", "y_ball"]] *= -1.0
        df = df[df["group_id"].isin([1, 2])].copy()
        for gid in (1, 2):
            all_pids[gid].update(df.loc[df["group_id"] == gid, "player_id"].unique().tolist())
        halves.append(df)

    g1_pids = sorted(all_pids[1])
    g2_pids = sorted(all_pids[2])
    pid2slot = {pid: i for i, pid in enumerate(g1_pids)}
    off = len(g1_pids)
    for i, pid in enumerate(g2_pids):
        pid2slot[pid] = off + i
    n_slots = len(pid2slot)
    team = np.array([0] * len(g1_pids) + [1] * len(g2_pids), dtype=np.int8)

    half_arrays = []
    for df in halves:
        ts = np.sort(df["timestamp_ms"].unique())
        ts_to_i = {t: i for i, t in enumerate(ts)}
        n10 = len(ts)

        ball10 = np.full((n10, 2), np.nan, dtype=np.float32)
        pos10 = np.full((n10, n_slots, 2), np.nan, dtype=np.float32)

        # Ball coordinates (take first row per timestamp after de-mirroring)
        b_df = df.groupby("timestamp_ms")[["x_ball", "y_ball"]].median().reindex(ts)
        ball10[:, 0] = b_df["x_ball"].to_numpy(dtype=np.float32) + hl_x
        ball10[:, 1] = b_df["y_ball"].to_numpy(dtype=np.float32) + hl_y

        # Labels with inactive precedence and unresolved conflict masking.
        grp = df.groupby("timestamp_ms").agg(
            alive_min=("AliveFlag", "min"),
            alive_max=("AliveFlag", "max"),
            poss_min=("PossessionFlag", "min"),
            poss_max=("PossessionFlag", "max"),
        ).reindex(ts)
        a_min = grp["alive_min"].to_numpy()
        a_max = grp["alive_max"].to_numpy()
        p_min = grp["poss_min"].to_numpy()
        p_max = grp["poss_max"].to_numpy()
        gt10 = np.full(n10, X, dtype=np.int8)
        # Conflict: AliveFlag disagrees OR both alive and PossessionFlag disagrees
        conflict_alive = a_min != a_max
        conflict_poss = (a_min == 1.0) & (p_min != p_max)
        # Give AliveFlag==0 precedence when alive_min==0 and alive_max==0
        gt10[a_max == 0.0] = D
        live_ok = (a_min == 1.0) & ~conflict_poss
        gt10[live_ok & (p_min == 1.0)] = H
        gt10[live_ok & (p_min == 2.0)] = A
        # Keep any conflicting row strictly X (ignore)
        gt10[conflict_alive | conflict_poss] = X

        t_idx = df["timestamp_ms"].map(ts_to_i).to_numpy()
        s_idx = df["player_id"].map(pid2slot).to_numpy()
        pos10[t_idx, s_idx, 0] = df["x"].to_numpy(dtype=np.float32) + hl_x
        pos10[t_idx, s_idx, 1] = df["y"].to_numpy(dtype=np.float32) + hl_y

        # Resample onto regular 40 ms (25 Hz) grid within [ts[0], ts[-1]]
        t25 = np.arange(ts[0], ts[-1] + 1, 40, dtype=np.int64)
        ball25 = _resample_10hz_to_25hz(ts, ball10, t25)
        pos25 = _resample_10hz_to_25hz(ts, pos10, t25)
        nearest_idx = np.clip(np.searchsorted(ts, t25), 0, n10 - 1)
        prev_idx = np.clip(nearest_idx - 1, 0, n10 - 1)
        pick = np.where(np.abs(t25 - ts[prev_idx]) < np.abs(t25 - ts[nearest_idx]), prev_idx, nearest_idx)
        gt25 = gt10[pick]
        # Mask any interpolated frame between conflicting/transition 10Hz ticks or >150ms gap
        bad_pair = (np.abs(ts[nearest_idx] - ts[prev_idx]) > 150)
        gt25[bad_pair] = X
        half_arrays.append((ball25, pos25, gt25))

    ball = np.concatenate([h[0] for h in half_arrays], axis=0)
    pos = np.concatenate([h[1] for h in half_arrays], axis=0)
    gt = np.concatenate([h[2] for h in half_arrays], axis=0)
    n1 = len(half_arrays[0][0])
    blocks = [(0, n1), (n1, len(ball))]
    spd = compute_speed(pos, blocks)

    m = Match(
        name="groningen",
        ball=ball,
        pos=pos,
        spd=spd,
        team=team,
        blocks=blocks,
        pitch=(pitch_l, pitch_w),
        gt=gt,
        meta={"source": "wcsf-groningen", "has_real_dead_ball": True, "orig_hz": 10},
    )
    _cache_save(cache_path, m)
    return m


def load_skillcorner_current(mid: str, force_rebuild: bool = False) -> Match:
    """Load one SkillCorner Open Data match @ 10 Hz -> 25 Hz.

    Only frames with valid possession.group ('home team'/'away team') AND finite
    players are labeled H/A; all other frames are labeled X (never fed as D,
    because SkillCorner omits player arrays when camera/replays leave play).
    """
    cache_path = os.path.join(CACHE_DIR, f"sc_cur_{mid}_25hz_v1.npz")
    if not force_rebuild and os.path.exists(cache_path):
        return _cache_load(cache_path)

    mdir = os.path.join(SC_CUR_DIR, str(mid))
    meta = json.load(open(os.path.join(mdir, f"{mid}_match.json")))
    pitch_l = float(meta.get("pitch_length") or 105.0)
    pitch_w = float(meta.get("pitch_width") or 68.0)
    hl_x, hl_y = 0.5 * pitch_l, 0.5 * pitch_w
    home_id = meta["home_team"]["id"]
    pid2team = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}

    rows_by_p: Dict[int, list] = {1: [], 2: []}
    with open(os.path.join(mdir, f"{mid}_tracking_extrapolated.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            per = o.get("period")
            if per in (1, 2):
                rows_by_p[per].append(o)

    # Order unique players present across tracking frames
    seen_pids = {0: set(), 1: set()}
    for per in (1, 2):
        for o in rows_by_p[per]:
            for pd_item in o.get("player_data") or []:
                pid = pd_item["player_id"]
                t = pid2team.get(pid)
                if t is not None:
                    seen_pids[t].add(pid)
    h_pids = sorted(seen_pids[0])
    a_pids = sorted(seen_pids[1])
    pid2slot = {pid: i for i, pid in enumerate(h_pids)}
    for i, pid in enumerate(a_pids):
        pid2slot[pid] = len(h_pids) + i
    n_slots = len(pid2slot)
    team = np.array([0] * len(h_pids) + [1] * len(a_pids), dtype=np.int8)

    half_arrays = []
    for per in (1, 2):
        objs = rows_by_p[per]
        objs.sort(key=lambda o: o["frame"])
        frames = np.array([int(o["frame"]) for o in objs], dtype=np.int64)
        ts10 = frames * 100  # 10 Hz -> 100 ms per frame
        n10 = len(objs)
        ball10 = np.full((n10, 2), np.nan, dtype=np.float32)
        pos10 = np.full((n10, n_slots, 2), np.nan, dtype=np.float32)
        gt10 = np.full(n10, X, dtype=np.int8)
        for i, o in enumerate(objs):
            bd = o.get("ball_data") or {}
            if bd.get("x") is not None and bd.get("y") is not None:
                ball10[i, 0] = float(bd["x"]) + hl_x
                ball10[i, 1] = float(bd["y"]) + hl_y
            plist = o.get("player_data") or []
            for pd_item in plist:
                s = pid2slot.get(pd_item["player_id"])
                if s is not None and pd_item.get("x") is not None and pd_item.get("y") is not None:
                    pos10[i, s, 0] = float(pd_item["x"]) + hl_x
                    pos10[i, s, 1] = float(pd_item["y"]) + hl_y
            grp = (o.get("possession") or {}).get("group")
            if len(plist) >= 14 and grp == "home team":
                gt10[i] = H
            elif len(plist) >= 14 and grp == "away team":
                gt10[i] = A

        t25 = np.arange(ts10[0], ts10[-1] + 1, 40, dtype=np.int64)
        ball25 = _resample_10hz_to_25hz(ts10, ball10, t25)
        pos25 = _resample_10hz_to_25hz(ts10, pos10, t25)
        nearest_idx = np.clip(np.searchsorted(ts10, t25), 0, n10 - 1)
        prev_idx = np.clip(nearest_idx - 1, 0, n10 - 1)
        pick = np.where(np.abs(t25 - ts10[prev_idx]) < np.abs(t25 - ts10[nearest_idx]), prev_idx, nearest_idx)
        gt25 = gt10[pick]
        # Mask if frame gap > 150ms or flanking 10Hz ticks disagree
        bad = (np.abs(ts10[nearest_idx] - ts10[prev_idx]) > 150) | (gt10[prev_idx] == X) | (gt10[nearest_idx] == X)
        gt25[bad] = X
        half_arrays.append((ball25, pos25, gt25))

    ball = np.concatenate([h[0] for h in half_arrays], axis=0)
    pos = np.concatenate([h[1] for h in half_arrays], axis=0)
    gt = np.concatenate([h[2] for h in half_arrays], axis=0)
    n1 = len(half_arrays[0][0])
    blocks = [(0, n1), (n1, len(ball))]
    spd = compute_speed(pos, blocks)

    m = Match(
        name=f"sc_{mid}",
        ball=ball,
        pos=pos,
        spd=spd,
        team=team,
        blocks=blocks,
        pitch=(pitch_l, pitch_w),
        gt=gt,
        meta={"source": "skillcorner-current", "has_real_dead_ball": False, "live_only": True},
    )
    _cache_save(cache_path, m)
    return m


def list_skillcorner_current() -> List[str]:
    if not os.path.isdir(SC_CUR_DIR):
        return []
    return sorted(d for d in os.listdir(SC_CUR_DIR) if os.path.isdir(os.path.join(SC_CUR_DIR, d)))


def load_all_skillcorner_current(force_rebuild: bool = False) -> List[Match]:
    return [load_skillcorner_current(mid, force_rebuild=force_rebuild) for mid in list_skillcorner_current()]


def _inject_burst_gaps(
    rng: np.random.Generator,
    n: int,
    blocks: List[Tuple[int, int]],
    target_miss: float,
    med_len: float = 200.0,
) -> np.ndarray:
    """Create contiguous True=missing masks distributed across blocks up to target_miss."""
    miss = np.zeros(n, dtype=bool)
    total_target = int(round(target_miss * sum(b - a for a, b in blocks)))
    cur = 0
    # Mix short optical dropouts (8..75f) and long occlusion/out-of-frame bursts (90..650f).
    attempts = 0
    while cur < total_target and attempts < 2000:
        attempts += 1
        a, b = blocks[int(rng.integers(0, len(blocks)))]
        if b - a < 20:
            continue
        if rng.random() < 0.45:
            L = int(rng.integers(8, 80))
        else:
            L = int(np.clip(rng.lognormal(mean=np.log(med_len), sigma=0.75), 80, 900))
        start = int(rng.integers(a, max(a + 1, b - L)))
        end = min(b, start + L)
        before = int(miss[start:end].sum())
        miss[start:end] = True
        cur += (end - start) - before
    return miss


def _smooth_drift(rng: np.random.Generator, shape: tuple, sigma_pos: float, sigma_white: float, corr_frames: float = 12.0) -> np.ndarray:
    """Correlated homography drift plus realistic low-amplitude per-frame noise."""
    from scipy.ndimage import gaussian_filter1d
    raw = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    drift = gaussian_filter1d(raw, sigma=corr_frames, axis=0)
    # Normalize variance of Gaussian-smoothed unit normal along axis 0
    std = float(np.std(drift)) or 1.0
    drift = drift * (sigma_pos / std)
    return drift + rng.normal(0.0, sigma_white, size=shape).astype(np.float32)


def degrade_atd(
    m: Match,
    scenario: str = "atd_real",
    seed: int = 0,
) -> Match:
    """Apply ATD-realistic tracking corruptions to an optical-tracking Match.

    Scenarios:
      - 'natural': uncorrupted clean tracking, unified speed recomputed.
      - 'atd_real': BAR-ATH-matched corruption (~40% total ball dropout combining
        live occlusion dropouts + higher stoppage dropout rate + residual real
        ball during ~30% of dead frames so visibility cannot trivially decide D;
        plus ~0.1% ball glitch teleports >40m/s, 20.5 avg visible players,
        tracklet fragmentation into 46 ATD slots with slot-ID swaps & 0.25m jitter).
      - 'atd_blind': adversarial corruption where ball dropout rate is UNIFORM
        40% across live and dead play AND 60% of dead frames still carry visible
        ball — proves the model works without exploiting ball-missing shortcuts.
      - 'atd_carried_dead': severe stress test where ball stays 100% VISIBLE during
        stoppages while 40% of live play loses the ball!
    """
    if scenario == "natural":
        return Match(
            name=m.name,
            ball=m.ball.copy(),
            pos=m.pos.copy(),
            spd=compute_speed(m.pos, m.blocks),
            team=m.team.copy(),
            blocks=list(m.blocks),
            pitch=m.pitch,
            gt=m.gt.copy() if m.gt is not None else None,
            meta=dict(m.meta, scenario="natural"),
        )

    rng = np.random.default_rng(seed)
    n, s_orig, _ = m.pos.shape
    gt = m.gt
    pitch_l, pitch_w = m.pitch

    # 1. Realistic player tracking noise, dropouts (~20.5 active), and tracklet fragmentation (46 ATD slots).
    n_slots = max(46, s_orig)
    h_idx = np.where(m.team == 0)[0]
    a_idx = np.where(m.team == 1)[0]
    n_h_slots = n_slots // 2
    n_a_slots = n_slots - n_h_slots
    new_team = np.array([0] * n_h_slots + [1] * n_a_slots, dtype=np.int8)
    new_pos = np.full((n, n_slots, 2), np.nan, dtype=np.float32)

    # Add realistic correlated spatial jitter (~0.20m wander + 0.025m noise) and short tracklet dropouts (~20.5/22 visible)
    p_pos = m.pos + _smooth_drift(rng, m.pos.shape, sigma_pos=0.20, sigma_white=0.025)
    for j in range(s_orig):
        drop = _inject_burst_gaps(rng, n, m.blocks, target_miss=0.065, med_len=35.0)
        p_pos[drop, j, :] = np.nan

    # Remap player identities into fragmented ATD tracklet slots every ~150-450 frames per player
    def _remap_team(orig_cols: np.ndarray, slot_offset: int, n_tslots: int) -> None:
        if len(orig_cols) == 0:
            return
        perm_slots = rng.permutation(n_tslots) + slot_offset
        cur_map = {c: perm_slots[i % n_tslots] for i, c in enumerate(orig_cols)}
        for a, b in m.blocks:
            t = a
            while t < b:
                step = int(rng.integers(125, 450))
                t2 = min(b, t + step)
                for c in orig_cols:
                    new_pos[t:t2, cur_map[c]] = p_pos[t:t2, c]
                # Swap one or two random player slots at segment boundary (simulate ID re-entry)
                used = set(cur_map.values())
                free = [slot_offset + k for k in range(n_tslots) if (slot_offset + k) not in used]
                if free and len(orig_cols) > 0:
                    c_swap = int(rng.choice(orig_cols))
                    cur_map[c_swap] = int(rng.choice(free))
                t = t2

    _remap_team(h_idx, 0, n_h_slots)
    _remap_team(a_idx, n_h_slots, n_a_slots)

    # 2. Ball corruption according to scenario.
    ball = m.ball.copy()
    is_dead = (gt == D) if gt is not None else np.zeros(n, dtype=bool)
    is_live = (gt == H) | (gt == A) if gt is not None else np.ones(n, dtype=bool)

    if scenario == "atd_real":
        # Match BAR-ATH ~40.7% missing rate (~70% of dead missed, ~16% of live missed, ~150 bursts)
        miss_live = _inject_burst_gaps(rng, n, m.blocks, target_miss=0.18, med_len=90.0) & is_live
        miss_dead = _inject_burst_gaps(rng, n, m.blocks, target_miss=0.72, med_len=280.0) & is_dead
        miss = miss_live | miss_dead
    elif scenario == "atd_blind":
        # Uniform ~41% burst missingness irrespective of live/dead (zeros out visibility leakage).
        miss = _inject_burst_gaps(rng, n, m.blocks, target_miss=0.41, med_len=220.0)
    elif scenario == "atd_carried_dead":
        # Severe adversarial: keep ball visible 100% during stoppages, drop ~40% during live play!
        miss = _inject_burst_gaps(rng, n, m.blocks, target_miss=0.40, med_len=140.0) & is_live
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    ball = ball + _smooth_drift(rng, ball.shape, sigma_pos=0.25, sigma_white=0.03)
    ball[miss] = np.nan

    # Inject rare 0.1% false ball teleport glitches (like ATD 54 m/s spikes) on visible frames
    vis_idx = np.where(np.isfinite(ball[:, 0]))[0]
    if len(vis_idx) > 0:
        n_glitch = max(1, int(0.001 * len(vis_idx)))
        g_idx = rng.choice(vis_idx, size=n_glitch, replace=False)
        ball[g_idx, 0] = rng.uniform(0.0, pitch_l, size=n_glitch).astype(np.float32)
        ball[g_idx, 1] = rng.uniform(0.0, pitch_w, size=n_glitch).astype(np.float32)

    spd = compute_speed(new_pos, m.blocks)
    return Match(
        name=m.name,
        ball=ball,
        pos=new_pos,
        spd=spd,
        team=new_team,
        blocks=list(m.blocks),
        pitch=m.pitch,
        gt=gt.copy() if gt is not None else None,
        meta=dict(m.meta, scenario=scenario, seed=seed),
    )


def augment_long_burst_gaps(
    m: Match,
    seed: int = 0,
    prob: float = 0.45,
    gap_range: Tuple[int, int] = (125, 750),
    retain_range: Tuple[int, int] = (200, 550),
) -> Match:
    """Inject 5-30s ball dropouts during training without changing benchmark presets."""
    if prob <= 0.0:
        return m
    rng = np.random.default_rng(seed)
    ball = m.ball.copy()
    g_lo, g_hi = int(gap_range[0]), int(gap_range[1])
    r_lo, r_hi = int(retain_range[0]), int(retain_range[1])
    for b0, b1 in m.blocks:
        if rng.random() > prob:
            continue
        cursor = b0 + int(rng.integers(r_lo, r_hi + 1))
        while cursor < b1:
            width = int(rng.integers(g_lo, g_hi + 1))
            ball[cursor : min(b1, cursor + width)] = np.nan
            cursor += width + int(rng.integers(r_lo, r_hi + 1))
    return Match(
        name=m.name,
        ball=ball,
        pos=m.pos,
        spd=m.spd,
        team=m.team,
        blocks=list(m.blocks),
        pitch=m.pitch,
        gt=m.gt,
        meta=dict(m.meta, long_burst_seed=seed),
    )


def fragment_player_tracks(
    pos: np.ndarray,
    team: np.ndarray,
    blocks: List[Tuple[int, int]],
    rng: np.random.Generator,
    min_slots: int = 46,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reassign complete team tracklets to spare slots without changing positions.

    This is shared by the opt-in unified Metrica degradation and does not alter
    historical benchmark presets. Team membership is preserved at every frame.
    """
    counts = [int(np.sum(team == t)) for t in (H, A)]
    slots_per_team = max((min_slots + 1) // 2, *counts)
    out = np.full((len(pos), 2 * slots_per_team, 2), np.nan, dtype=np.float32)
    new_team = np.repeat(np.array([H, A], dtype=np.int8), slots_per_team)
    for code in (H, A):
        cols = np.flatnonzero(team == code)
        offset = code * slots_per_team
        for a, b in blocks:
            mapping = rng.permutation(slots_per_team)[:len(cols)] + offset
            cursor = a
            while cursor < b:
                stop = min(b, cursor + int(rng.integers(125, 451)))
                out[cursor:stop, mapping] = pos[cursor:stop, cols]
                free = np.setdiff1d(np.arange(offset, offset + slots_per_team), mapping)
                if len(cols) and len(free):
                    mapping[int(rng.integers(len(cols)))] = int(rng.choice(free))
                cursor = stop
    return out, new_team


def augment_stoppage_ball(m: Match, seed: int = 0) -> Match:
    """External-only ball-motion augmentation to break frozen-stoppage shortcuts.

    Randomly hide, carry, or smoothly drift the ball in each labelled stoppage.
    Fixed 21-frame, degree-2 Savitzky-Golay smoothing on finite ball runs also
    exposes ownership to smoothed tracking. Player arrays and labels are kept.
    This is a training recipe, never an evaluation scenario or inference fit.
    """
    if m.gt is None:
        raise ValueError("Stoppage augmentation requires labelled external data")
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(seed)
    ball = m.ball.copy()
    for a, b in m.blocks:
        dead = m.gt[a:b] == D
        edges = np.diff(np.r_[False, dead, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            s, e = a + int(start), a + int(stop)
            mode = int(rng.integers(3))
            if mode == 0:
                ball[s:e] = np.nan
                continue
            before = ball[s - 1] if s > a else np.array([np.nan, np.nan])
            after = ball[e] if e < b else np.array([np.nan, np.nan])
            anchor = before if np.isfinite(before).all() else after
            if not np.isfinite(anchor).all():
                anchor = np.asarray(m.pitch, dtype=np.float32) / 2
            if mode == 1:
                ball[s:e] = anchor
            else:
                end = after if np.isfinite(after).all() else anchor
                u = np.linspace(0.0, 1.0, e - s, dtype=np.float32)[:, None]
                excursion = rng.normal(0.0, 2.0, size=2).astype(np.float32)
                ball[s:e] = (1 - u) * anchor + u * end + np.sin(np.pi * u) * excursion
        finite = np.isfinite(ball[a:b]).all(axis=1)
        edges = np.diff(np.r_[False, finite, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            s, e = a + int(start), a + int(stop)
            width = min(21, (e - s - 1) // 2 * 2 + 1)
            if width >= 5:
                ball[s:e] = savgol_filter(ball[s:e], width, 2, axis=0, mode="interp")
    return Match(name=m.name, ball=ball, pos=m.pos, spd=m.spd, team=m.team,
                 blocks=list(m.blocks), pitch=m.pitch, gt=m.gt,
                 meta=dict(m.meta, stoppage_augmentation_seed=int(seed)))
