"""Guardrails for the ball-gated dead-ball onset snap."""
import unittest

import numpy as np

from solution import line_snap
from solution.common import A, D, H

PITCH = (105.0, 68.0)


def _case():
    n = 400
    codes = np.full(n, H, dtype=np.int8)
    codes[200:300] = D          # predicted stoppage, onset late at 200
    codes[300:] = A
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    t = np.arange(n)
    ball[:, 0] = 50.0
    ball[:, 1] = np.clip(60.0 + 0.1 * (t - 100), None, 80.0)   # crosses y = 68.3 at t = 183
    return codes, ball, [(0, n)]


def _spells(codes):
    dead = codes == D
    return int(np.sum(dead[1:] & ~dead[:-1]) + dead[0])


class LineSnapTests(unittest.TestCase):
    def test_moves_late_onset_to_observed_exit(self):
        codes, ball, blocks = _case()
        out = line_snap.snap_onsets(codes, ball, PITCH, blocks)
        first_dead = int(np.flatnonzero(out == D)[0])
        self.assertEqual(first_dead, 184 + line_snap.DEFAULTS["delay"])
        self.assertEqual(_spells(out), _spells(codes))
        np.testing.assert_array_equal(out[300:], codes[300:])

    def test_no_ball_no_change(self):
        codes, ball, blocks = _case()
        ball[:] = np.nan
        np.testing.assert_array_equal(line_snap.snap_onsets(codes, ball, PITCH, blocks), codes)

    def test_never_creates_a_stoppage(self):
        codes, ball, blocks = _case()
        codes[:] = H
        np.testing.assert_array_equal(line_snap.snap_onsets(codes, ball, PITCH, blocks), codes)

    def test_teleport_glitch_rejected(self):
        codes, ball, blocks = _case()
        ball[:, 1] = 30.0
        ball[150, :] = (50.0, 90.0)   # single-frame jump outside the pitch
        np.testing.assert_array_equal(line_snap.snap_onsets(codes, ball, PITCH, blocks), codes)

    def test_team_swap_equivariance(self):
        codes, ball, blocks = _case()
        swap = np.where(codes == H, A, np.where(codes == A, H, codes)).astype(np.int8)
        out = line_snap.snap_onsets(codes, ball, PITCH, blocks)
        out_sw = line_snap.snap_onsets(swap, ball, PITCH, blocks)
        expect = np.where(out == H, A, np.where(out == A, H, out))
        np.testing.assert_array_equal(out_sw, expect)


if __name__ == "__main__":
    unittest.main()
