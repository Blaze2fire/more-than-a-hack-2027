"""python -m solution.tests.test_core"""
import itertools

import numpy as np

from solution import external, model
from solution.common import A, D, H, X


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
