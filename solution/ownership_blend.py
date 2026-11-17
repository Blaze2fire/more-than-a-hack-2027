"""Blend conditional live ownership while preserving the dead log-probability exactly."""
import numpy as np

from solution.common import A, D, H


def blend_logp(lp_src: np.ndarray, p_ha_nn: np.ndarray, w_nn: float) -> np.ndarray:
    if not 0.0 <= w_nn <= 1.0:
        raise ValueError("Ownership blend weight must be in [0, 1]")
    if lp_src.ndim != 2 or lp_src.shape[1] != 3 or p_ha_nn.shape != (len(lp_src),):
        raise ValueError("Expected (N, 3) log probabilities and (N,) ownership probabilities")
    if not np.isfinite(lp_src).all() or not np.isfinite(p_ha_nn).all():
        raise ValueError("Blend inputs must be finite")
    if ((p_ha_nn < 0) | (p_ha_nn > 1)).any():
        raise ValueError("Ownership probabilities must be in [0, 1]")
    if w_nn == 0:
        return lp_src.copy()
    p = np.exp(lp_src.astype(np.float64))
    # The centered difference is exactly antisymmetric under H/A exchange.
    d_base = (p[:, H] - p[:, A]) / np.maximum(p[:, H] + p[:, A], 1e-12)
    d_mix = (1.0 - w_nn) * d_base + w_nn * (2.0 * p_ha_nn.astype(np.float64) - 1.0)
    d_q = np.round(np.clip(d_mix, -0.9998, 0.9998) * 65536.0) / 65536.0
    live = 1.0 - p[:, D]
    out = lp_src.copy()
    out[:, H] = np.log(np.clip(live * (0.5 + 0.5 * d_q), 1e-5, 1.0))
    out[:, A] = np.log(np.clip(live * (0.5 - 0.5 * d_q), 1e-5, 1.0))
    # D is retained from lp_src, without an exp/log round trip or new clipping.
    return out
