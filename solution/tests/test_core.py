"""python -m solution.tests.test_core"""
import dataclasses
import itertools

import numpy as np

from solution import ball_tracker, external, features_v4, model, pipeline_v4, temporal, viterbi4
from solution.common import A, D, H, Match, X


def brute(s, c_team, c_dead, c_live):
    n = len(s[0])
    best, arg = -1e18, None
    for path in itertools.product(range(3), repeat=n):
        v = sum(s[k][t] for t, k in enumerate(path))
        for a, b in zip(path, path[1:]):
            if a != b:
                v -= c_dead if b == 2 else (c_live if a == 2 else c_team)
        if v > best:
            best, arg = v, path
    return best


def score(path, s, c_team, c_dead, c_live):
    v = sum(s[k][t] for t, k in enumerate(path))
    for a, b in zip(path, path[1:]):
        if a != b:
            v -= c_dead if b == 2 else (c_live if a == 2 else c_team)
    return v


def test_viterbi_optimal():
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = rng.normal(0, 1, (3, 7))
        costs = rng.uniform(0.2, 2.0, 3)
        path = model.viterbi3(s[0], s[1], s[2], *costs)
        assert abs(score(path, s, *costs) - brute(s, *costs)) < 1e-9


def test_absorb_short():
    codes = np.array([H] * 10 + [A] * 3 + [H] * 10 + [D] * 2 + [A] * 10, dtype=np.int8)
    out = model.absorb_short(codes, 5)
    assert (out[:23] == H).all() and (out[23:25] == D).all() and (out[25:] == A).all()
    # a short live spell right after dead ball joins the next live spell, never the dead ball
    codes = np.array([D] * 10 + [H] * 3 + [A] * 10, dtype=np.int8)
    out = model.absorb_short(codes, 5)
    assert (out[:10] == D).all() and (out[10:] == A).all()
    # dead ball on both sides: left alone
    codes = np.array([D] * 10 + [H] * 3 + [D] * 10, dtype=np.int8)
    assert (model.absorb_short(codes, 5) == codes).all()


def test_derive_gt():
    n = 400
    missing = np.zeros(n, dtype=bool)
    missing[:20] = True            # before kickoff
    missing[200:260] = True        # ball out of play
    events = [(20, 0, "SET PIECE"), (25, 0, "PASS"), (100, 1, "RECOVERY"), (110, 0, "RECOVERY"),
              (120, 0, "PASS"), (150, 1, "RECOVERY"), (160, 1, "PASS"), (260, 1, "SET PIECE")]
    gt = external.derive_gt(n, [(0, n)], missing, sorted(events))
    assert (gt[:20] == D).all()
    assert (gt[20:150] == H).all()     # the 10-frame away touch at 100 is uncontrolled -> merged
    assert (gt[150:200] == A).all()
    assert (gt[200:260] == D).all() and (gt[260:] == A).all()


def _brute4_best(sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias):
    n = len(sh)
    best = -1e18
    for path in itertools.product(range(4), repeat=n):
        v = sum((sd[t] + d_bias if k >= 2 else (sh[t] if k == 0 else sa[t])) for t, k in enumerate(path))
        ok = True
        for a, b in zip(path, path[1:]):
            if a == b:
                continue
            if (a, b) in ((0, 1), (1, 0)):
                v -= c_team
            elif (a, b) in ((0, 2), (1, 3)):
                v -= c_stop
            elif (a, b) in ((2, 0), (3, 1)):
                v -= c_res_same
            elif (a, b) in ((2, 1), (3, 0)):
                v -= c_res_flip
            else:
                ok = False
                break
        if ok and v > best:
            best = v
    return best


def _score4_collapsed(codes, sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias):
    """Best valid 4-state DP score restricted to collapsed state sequence `codes`."""
    INF = 1e18
    n = len(codes)
    dp = np.full(4, -INF)
    if codes[0] == H:
        dp[0] = sh[0]
    elif codes[0] == A:
        dp[1] = sa[0]
    else:
        dp[2] = sd[0] + d_bias
        dp[3] = sd[0] + d_bias
    trans = {
        (0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0, (3, 3): 0.0,
        (0, 1): -c_team, (1, 0): -c_team,
        (0, 2): -c_stop, (1, 3): -c_stop,
        (2, 0): -c_res_same, (3, 1): -c_res_same,
        (2, 1): -c_res_flip, (3, 0): -c_res_flip,
    }
    for t in range(1, n):
        allowed = (0,) if codes[t] == H else ((1,) if codes[t] == A else (2, 3))
        ndp = np.full(4, -INF)
        em = sh[t] if codes[t] == H else (sa[t] if codes[t] == A else (sd[t] + d_bias))
        for cur in allowed:
            cands = [dp[prv] + cost for (prv, c), cost in trans.items() if c == cur and dp[prv] > -INF / 2]
            if cands:
                ndp[cur] = max(cands) + em
        dp = ndp
    return float(dp.max())


def test_viterbi4_optimal():
    rng = np.random.default_rng(1)
    for _ in range(30):
        sh, sa, sd = rng.normal(0, 1, (3, 6))
        c_team, c_stop, c_res_same, c_res_flip = rng.uniform(0.3, 3.0, 4)
        d_bias = float(rng.uniform(-0.4, 0.4))
        path = viterbi4.viterbi4_block(sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias=d_bias)
        b = _brute4_best(sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias)
        s = _score4_collapsed(path, sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias)
        assert abs(s - b) < 1e-9, (s, b)


def test_repair_spells_block():
    c = np.array([H] * 20 + [D] * 5 + [H] * 20, dtype=np.int8)
    out = viterbi4.repair_spells_block(c, min_dead=10, min_live_blip=0, min_turn=0)
    assert (out == H).all()

    c = np.array([D] * 20 + [H] * 6 + [D] * 20, dtype=np.int8)
    out = viterbi4.repair_spells_block(c, min_dead=0, min_live_blip=10, min_turn=0)
    assert (out == D).all()

    c = np.array([H] * 20 + [A] * 5 + [H] * 20, dtype=np.int8)
    out = viterbi4.repair_spells_block(c, min_dead=0, min_live_blip=0, min_turn=10)
    assert (out == H).all()

    c = np.array([H] * 20 + [A] * 15 + [H] * 20, dtype=np.int8)
    out = viterbi4.repair_spells_block(c, min_dead=0, min_live_blip=0, min_turn=10)
    assert (out[20:35] == A).all()


def _make_synth_match(seed: int = 0) -> Match:
    rng = np.random.default_rng(seed)
    n = 600
    blocks = [(0, 300), (300, 600)]
    t = np.linspace(0.0, 24.0, n, dtype=np.float32)
    team = np.array([H] * 11 + [A] * 11, dtype=np.int8)
    pos = np.zeros((n, 22, 2), dtype=np.float32)
    base_x = rng.uniform(15.0, 90.0, 22).astype(np.float32)
    base_y = rng.uniform(10.0, 58.0, 22).astype(np.float32)
    pos[:, :, 0] = base_x[None, :] + 3.0 * np.sin(t[:, None] + rng.uniform(0, 6, 22))
    pos[:, :, 1] = base_y[None, :] + 2.0 * np.cos(0.7 * t[:, None] + rng.uniform(0, 6, 22))
    ball = np.column_stack([52.5 + 18.0 * np.sin(0.8 * t), 34.0 + 12.0 * np.cos(0.6 * t)]).astype(np.float32)
    ball[60:95] = np.nan
    ball[380:415] = np.nan
    spd = np.zeros((n, 22), dtype=np.float32)
    gt = np.full(n, H, dtype=np.int8)
    gt[100:200] = A
    gt[200:260] = D
    gt[260:300] = H
    gt[300:400] = A
    gt[400:480] = D
    gt[480:600] = H
    return Match("synth", ball=ball, pos=pos, spd=spd, team=team, blocks=blocks, pitch=(105.0, 68.0), gt=gt)


def test_dead_zero_ball_leakage():
    m1 = _make_synth_match(0)
    m2 = dataclasses.replace(m1, ball=np.full_like(m1.ball, np.nan))
    m3 = dataclasses.replace(m1, ball=(-m1.ball + 80.0))
    X1 = features_v4.build_dead_features(m1)
    X2 = features_v4.build_dead_features(m2)
    X3 = features_v4.build_dead_features(m3)
    assert float(np.nanmax(np.abs(X1 - X2))) == 0.0
    assert float(np.nanmax(np.abs(X1 - X3))) == 0.0
    pd = np.linspace(0.05, 0.95, m1.n, dtype=np.float32)
    Z1 = temporal.build_dead_seq_features(pd, X1[:, :40], m1.blocks)
    Z2 = temporal.build_dead_seq_features(pd, X2[:, :40], m2.blocks)
    assert float(np.nanmax(np.abs(Z1 - Z2))) == 0.0


def test_team_and_pipeline_swap_equivariance():
    m = _make_synth_match(1)
    m_sw = dataclasses.replace(
        m, team=np.where(m.team == H, A, np.where(m.team == A, H, m.team)).astype(np.int8)
    )
    imp = ball_tracker.fit_constellation_imputer([m], stride=2, seed=0)
    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    vel_sw = ball_tracker.compute_player_vel(m_sw.pos, m_sw.blocks)
    rb = ball_tracker.reconstruct_ball(m, imputer=imp, vel=vel)
    rb_sw = ball_tracker.reconstruct_ball(m_sw, imputer=imp, vel=vel_sw)
    assert float(np.nanmax(np.abs(rb.ball_hat - rb_sw.ball_hat))) < 1e-6

    Xtf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False)
    Xts = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True)
    Xtf_sw = features_v4.build_team_features(m_sw, rb=rb_sw, vel=vel_sw, swap=False)
    assert float(np.nanmax(np.abs(Xts - Xtf_sw))) < 1e-5

    p_d_test = np.full(m.n, 0.1, dtype=np.float32)
    p_ha_test = np.round(np.clip(0.5 + 0.25 * np.sin(np.arange(m.n, dtype=np.float64)), 0.05, 0.95) * 65536.0) / 65536.0
    p_ha_f32 = p_ha_test.astype(np.float32)
    p_ha_sw_f32 = (1.0 - p_ha_test).astype(np.float32)
    Zts = temporal.build_team_seq_features(p_ha_f32, p_d_test, Xtf[:, :65], m.blocks, swap=True, Xs_base=Xts[:, :65])
    Ztf_sw = temporal.build_team_seq_features(p_ha_sw_f32, p_d_test, Xtf_sw[:, :65], m_sw.blocks, swap=False)
    assert float(np.max(np.abs(Zts - Ztf_sw))) < 1e-5

    Xd = features_v4.build_dead_features(m, vel=vel)
    dm = m.gt != X
    lm = (m.gt == H) | (m.gt == A)
    ds1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=0, use_gpu=False).fit(Xd[dm], (m.gt[dm] == D).astype(np.int8))
    ts1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=1, use_gpu=False).fit(
        np.vstack([Xtf[lm], Xts[lm]]),
        np.concatenate([(m.gt[lm] == H).astype(np.int8), (m.gt[lm] == A).astype(np.int8)]),
    )
    pd1 = np.clip(ds1.predict_proba(Xd)[:, 1], 1e-4, 1.0 - 1e-4).astype(np.float32)
    ph = ts1.predict_proba(Xtf)[:, 1].astype(np.float64)
    pa = ts1.predict_proba(Xts)[:, 1].astype(np.float64)
    d_ha = (ph - pa) / np.maximum(ph + pa, 1e-6)
    d_ha_q = np.round(np.clip(d_ha, -0.9998, 0.9998) * 65536.0) / 65536.0
    pha1 = np.clip(0.5 + 0.5 * d_ha_q, 1e-4, 1.0 - 1e-4).astype(np.float32)
    stk = pipeline_v4.fit_temporal_stacker_from_oof(
        [(pd1, pha1, Xd[:, :40], Xtf[:, :65], Xts[:, :65], m)],
        stride=1,
        seed=0,
        mode="lgbm",
        use_gpu=False,
    )
    pipe = pipeline_v4.PipelineV4(imputer=imp, dead_s1=ds1, team_s1=ts1, stacker=stk)
    lp = pipe.predict_log_proba(m)
    lp_sw = pipe.predict_log_proba(m_sw)
    assert float(np.max(np.abs(lp[:, [A, H, D]] - lp_sw))) < 1e-5


def _brute4_dynamic_best(sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias, sw_sig, c_sw_gain):
    n = len(sh)
    ct_arr = np.maximum(0.35 * c_team, c_team - c_sw_gain * sw_sig)
    best = -1e18
    for path in itertools.product(range(4), repeat=n):
        v = sum((sd[t] + d_bias if k >= 2 else (sh[t] if k == 0 else sa[t])) for t, k in enumerate(path))
        ok = True
        for t in range(1, n):
            a, b = path[t - 1], path[t]
            if a == b:
                continue
            if (a, b) in ((0, 1), (1, 0)):
                v -= float(ct_arr[t])
            elif (a, b) in ((0, 2), (1, 3)):
                v -= c_stop
            elif (a, b) in ((2, 0), (3, 1)):
                v -= c_res_same
            elif (a, b) in ((2, 1), (3, 0)):
                v -= c_res_flip
            else:
                ok = False
                break
        if ok and v > best:
            best = v
    return best


def test_viterbi4_dynamic_sw_optimal():
    rng = np.random.default_rng(7)
    for _ in range(25):
        sh, sa, sd = rng.normal(0, 1, (3, 6))
        sw_sig = rng.uniform(0.0, 1.0, 6).astype(np.float32)
        c_team = float(rng.uniform(1.5, 4.0))
        c_sw_gain = float(rng.uniform(0.2, 0.7 * c_team))
        c_stop, c_res_same, c_res_flip = rng.uniform(0.5, 3.0, 3)
        d_bias = float(rng.uniform(-0.3, 0.3))
        path = viterbi4.viterbi4_block(
            sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias=d_bias, sw_signal=sw_sig, c_sw_gain=c_sw_gain
        )
        # Verify exact DP score of recovered path equals brute-force optimum
        b = _brute4_dynamic_best(sh, sa, sd, c_team, c_stop, c_res_same, c_res_flip, d_bias, sw_sig, c_sw_gain)
        ct_arr = np.maximum(0.35 * c_team, c_team - c_sw_gain * sw_sig)
        INF = 1e18
        dp = np.full(4, -INF)
        if path[0] == H:
            dp[0] = sh[0]
        elif path[0] == A:
            dp[1] = sa[0]
        else:
            dp[2] = dp[3] = sd[0] + d_bias
        for t in range(1, len(path)):
            allowed = (0,) if path[t] == H else ((1,) if path[t] == A else (2, 3))
            ndp = np.full(4, -INF)
            em = sh[t] if path[t] == H else (sa[t] if path[t] == A else (sd[t] + d_bias))
            trans = {
                (0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0, (3, 3): 0.0,
                (0, 1): -float(ct_arr[t]), (1, 0): -float(ct_arr[t]),
                (0, 2): -c_stop, (1, 3): -c_stop,
                (2, 0): -c_res_same, (3, 1): -c_res_same,
                (2, 1): -c_res_flip, (3, 0): -c_res_flip,
            }
            for cur in allowed:
                cands = [dp[prv] + cost for (prv, c), cost in trans.items() if c == cur and dp[prv] > -INF / 2]
                if cands:
                    ndp[cur] = max(cands) + em
            dp = ndp
        assert abs(float(dp.max()) - b) < 1e-8


def test_centroid_axis_zero_leak_and_swap():
    m1 = _make_synth_match(2)
    m2 = dataclasses.replace(m1, ball=np.full_like(m1.ball, np.nan))
    vel1 = ball_tracker.compute_player_vel(m1.pos, m1.blocks)
    vel2 = ball_tracker.compute_player_vel(m2.pos, m2.blocks)
    c1 = features_v4.compute_centroid_axis_motion(m1, vel=vel1, swap=False)
    c2 = features_v4.compute_centroid_axis_motion(m2, vel=vel2, swap=False)
    assert c1.shape == (m1.n, 45)
    assert float(np.max(np.abs(c1 - c2))) == 0.0

    m_sw = dataclasses.replace(
        m1, team=np.where(m1.team == H, A, np.where(m1.team == A, H, m1.team)).astype(np.int8)
    )
    c_s = features_v4.compute_centroid_axis_motion(m1, vel=vel1, swap=True)
    c_sw = features_v4.compute_centroid_axis_motion(m_sw, vel=vel1, swap=False)
    assert float(np.max(np.abs(c_s - c_sw))) < 1e-6


def test_neural_team_swap_and_dead_invariance():
    from solution import neural_team
    m = _make_synth_match(3)
    m_sw = dataclasses.replace(
        m, team=np.where(m.team == H, A, np.where(m.team == A, H, m.team)).astype(np.int8)
    )
    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    rb = ball_tracker.reconstruct_ball(m, imputer=None, vel=vel)
    Xtf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False, use_p3=True)
    Xts = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True, use_p3=True)
    wins = neural_team.extract_match_neural_windows(m, rb, vel, Xtf, Xts, win_len=200, max_wins=2)
    mdl = neural_team.fit_neural_team_from_windows(wins, epochs=2, seed=0, use_p3=True, use_gpu=False)

    pha, psw = neural_team.predict_neural_team(mdl, m, rb=rb, vel=vel, Xtf=Xtf, Xts=Xts, use_gpu=False)
    rb_sw = ball_tracker.reconstruct_ball(m_sw, imputer=None, vel=vel)
    Xtf_sw = features_v4.build_team_features(m_sw, rb=rb_sw, vel=vel, swap=False, use_p3=True)
    Xts_sw = features_v4.build_team_features(m_sw, rb=rb_sw, vel=vel, swap=True, use_p3=True)
    pha_sw, psw_sw = neural_team.predict_neural_team(mdl, m_sw, rb=rb_sw, vel=vel, Xtf=Xtf_sw, Xts=Xts_sw, use_gpu=False)

    assert float(np.max(np.abs((1.0 - pha) - pha_sw))) < 1e-4
    assert float(np.max(np.abs(psw - psw_sw))) < 1e-4

    # Check dead_leak == 0.0 in PipelineV4 when neural_model and bf specialist are present
    Xd = features_v4.build_dead_features(m, vel=vel)
    dm = m.gt != X
    lm = (m.gt == H) | (m.gt == A)
    bf_cols = features_v4.ball_free_team_cols(Xtf.shape[1])
    ds1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=0, use_gpu=False).fit(Xd[dm], (m.gt[dm] == D).astype(np.int8))
    ts1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=1, use_gpu=False).fit(
        np.vstack([Xtf[lm], Xts[lm]]),
        np.concatenate([(m.gt[lm] == H).astype(np.int8), (m.gt[lm] == A).astype(np.int8)]),
    )
    tbf = pipeline_v4.EnsembleBinaryClf("lgbm", seed=2, use_gpu=False).fit(
        np.vstack([Xtf[lm][:, bf_cols], Xts[lm][:, bf_cols]]),
        np.concatenate([(m.gt[lm] == H).astype(np.int8), (m.gt[lm] == A).astype(np.int8)]),
    )
    pipe = pipeline_v4.PipelineV4(
        imputer=None, dead_s1=ds1, team_s1=ts1, stacker=None,
        team_bf_s1=tbf, bf_cols=bf_cols, neural_model=mdl, neural_weight=0.35, use_p3=True,
    )
    m_nan = dataclasses.replace(m, ball=np.full_like(m.ball, np.nan))
    lp = pipe.predict_log_proba(m)
    lp_nan = pipe.predict_log_proba(m_nan)
    assert float(np.max(np.abs(lp[:, D] - lp_nan[:, D]))) == 0.0
    lp_p_sw = pipe.predict_log_proba(m_sw)
    assert float(np.max(np.abs(lp[:, [A, H, D]] - lp_p_sw))) < 1e-4


def test_neural_v2_exact_swap_and_dead_invariance():
    import torch
    from solution import neural_v2
    m = _make_synth_match(4)
    m_sw = dataclasses.replace(
        m, team=np.where(m.team == H, A, np.where(m.team == A, H, m.team)).astype(np.int8)
    )
    torch.manual_seed(0)
    net = neural_v2.OwnNet(width=32, pdim=16, gru_layers=1)
    mdl = [neural_v2.NeuralV2Model(state_dict=net.state_dict(), width=32, pdim=16, gru_layers=1)]
    pha, psw = neural_v2.predict_match(mdl, m)
    pha_sw, psw_sw = neural_v2.predict_match(mdl, m_sw)
    assert float(np.std(pha)) > 0.0
    assert float(np.max(np.abs((1.0 - pha.astype(np.float64)) - pha_sw))) == 0.0
    assert float(np.max(np.abs(psw - psw_sw))) == 0.0

    vel = ball_tracker.compute_player_vel(m.pos, m.blocks)
    Xd = features_v4.build_dead_features(m, vel=vel)
    rb = ball_tracker.reconstruct_ball(m, imputer=None, vel=vel)
    Xtf = features_v4.build_team_features(m, rb=rb, vel=vel, swap=False)
    Xts = features_v4.build_team_features(m, rb=rb, vel=vel, swap=True)
    dm = m.gt != X
    lm = (m.gt == H) | (m.gt == A)
    ds1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=0, use_gpu=False).fit(Xd[dm], (m.gt[dm] == D).astype(np.int8))
    ts1 = pipeline_v4.EnsembleBinaryClf("lgbm", seed=1, use_gpu=False).fit(
        np.vstack([Xtf[lm], Xts[lm]]),
        np.concatenate([(m.gt[lm] == H).astype(np.int8), (m.gt[lm] == A).astype(np.int8)]),
    )
    pipe = pipeline_v4.PipelineV4(imputer=None, dead_s1=ds1, team_s1=ts1, stacker=None, neural_v2=mdl, neural_weight=0.7)
    lp = pipe.predict_log_proba(m)
    lp_nan = pipe.predict_log_proba(dataclasses.replace(m, ball=np.full_like(m.ball, np.nan)))
    assert float(np.max(np.abs(lp[:, D] - lp_nan[:, D]))) == 0.0
    assert float(np.max(np.abs(lp[:, [A, H, D]] - pipe.predict_log_proba(m_sw)))) < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
