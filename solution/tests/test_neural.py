"""Guardrails for the ownership-only sequence branch. No competition files are read."""
import unittest

import numpy as np

from solution import neural_v2
from solution.common import A, D, H
from solution.ownership_blend import blend_logp


class OwnershipGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        neural_v2.torch.set_num_threads(1)

    def test_blend_retains_dead_exactly_and_swaps(self):
        rng = np.random.default_rng(19)
        p = rng.dirichlet([0.2, 0.2, 0.2], 1024)
        lp = np.log(np.clip(p, 1e-10, 1)).astype(np.float32)
        nn = (rng.integers(10, 65526, 1024) / 65536).astype(np.float32)
        for weight in (0, 0.3, 0.7, 1):
            mixed = blend_logp(lp, nn, weight)
            np.testing.assert_array_equal(mixed[:, D], lp[:, D])
            swapped = blend_logp(lp[:, [A, H, D]], 1 - nn, weight)
            np.testing.assert_array_equal(swapped, mixed[:, [A, H, D]])
        np.testing.assert_array_equal(blend_logp(lp, nn, 0), lp)

    def test_neural_inference_swap_and_missing_ball(self):
        rng = np.random.default_rng(2)
        n = 80
        P = rng.normal(size=(n, 2, 11, 4)).astype(np.float16)
        M = (rng.random((n, 2, 11)) > 0.15)
        ball = rng.normal(size=(n, 2)).astype(np.float32)
        ball[20:50] = np.nan
        c = dict(P=P, M=M, ball=ball, blocks=[(3, 37), (41, 80)])
        for stride in (1, 5):
            neural_v2.torch.manual_seed(1)
            net = neural_v2.OwnNet(width=16, pdim=8, gru_layers=1, frame_stride=stride)
            model = neural_v2.NeuralV2Model(net.state_dict(), width=16, pdim=8, gru_layers=1,
                                             frame_stride=stride)
            p, q = neural_v2.predict_compact([model], c, chunk=20, halo=20)
            swapped = dict(c, P=P[:, ::-1].copy(), M=M[:, ::-1].copy())
            ps, qs = neural_v2.predict_compact([model], swapped, chunk=20, halo=20)
            np.testing.assert_array_equal(ps, 1 - p)
            np.testing.assert_array_equal(qs, q)
            empty_ball = dict(c, ball=np.full_like(ball, np.nan))
            ep, eq = neural_v2.predict_compact([model], empty_ball, chunk=20, halo=20)
            self.assertTrue(np.isfinite(ep).all() and np.isfinite(eq).all())
            outside = np.ones(n, dtype=bool)
            for a, b in c["blocks"]:
                outside[a:b] = False
            changed_ball, changed_P = ball.copy(), P.copy()
            changed_ball[outside], changed_P[outside] = 1000, -1000
            cp, cq = neural_v2.predict_compact([model], dict(c, P=changed_P, ball=changed_ball), chunk=20, halo=20)
            np.testing.assert_array_equal(cp, p)
            np.testing.assert_array_equal(cq, q)

    def test_player_set_permutation(self):
        torch = neural_v2.torch
        torch.manual_seed(8)
        net = neural_v2.OwnNet(width=16, pdim=8, gru_layers=1).eval()
        P = torch.randn(1, 30, 2, 11, 4)
        M = (torch.rand(1, 30, 2, 11) > 0.1).float()
        ball = torch.randn(1, 30, 2)
        perm = torch.randperm(11)
        with torch.no_grad():
            own, switch = net(P, M, ball)
            own2, switch2 = net(P[..., perm, :], M[..., perm], ball)
        torch.testing.assert_close(own, own2, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(switch, switch2, atol=2e-6, rtol=2e-6)

    def test_sampling_rate_preserves_physical_ball_velocity(self):
        torch = neural_v2.torch
        raw = torch.zeros((1, 100, 2))
        raw[0, :, 0] = torch.arange(100) * (10 / 25)
        P = torch.zeros((1, 100, 2, 11, 4))
        M = torch.ones((1, 100, 2, 11))
        *_, context25 = neural_v2.featurize(P, M, raw, fps=25)
        *_, context5 = neural_v2.featurize(P[:, ::5], M[:, ::5], raw[:, ::5], fps=5)
        np.testing.assert_allclose(context25[0, 10:85:5, 7].numpy(), context5[0, 2:17, 7].numpy(), atol=1e-6)

    def test_gap_policies(self):
        gt = np.full(1500, H, dtype=np.int8)
        np.testing.assert_array_equal(neural_v2.sample_gap_mask(np.random.default_rng(1), gt, "none"), False)
        masks = [neural_v2.sample_gap_mask(np.random.default_rng(k), gt, "mixed") for k in range(40)]
        longest = 0
        for mask in masks:
            bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            if len(bounds):
                longest = max(longest, int(np.max(bounds[1::2] - bounds[::2])))
        self.assertGreaterEqual(longest, 5 * 25)
        with self.assertRaises(ValueError):
            neural_v2.sample_gap_mask(np.random.default_rng(1), gt, "invalid")


if __name__ == "__main__":
    unittest.main()
