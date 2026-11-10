"""Finite-difference check of the hand-written gradients in bhi.channel.

Run:  python -m tests.test_gradients     (from the repository root)
"""
import numpy as np
from bhi import channel as ch


def check(lam, mu, seed=0, eps=1e-6, n_checks=40):
    rng = np.random.default_rng(seed)
    d, n, H, sigma = 7, 5, 9, 0.1
    P = ch.init_params(d, n, H, rng)
    for k in ("a", "b", "c"):
        P[k] = rng.normal(0, 0.3, P[k].shape)          # avoid symmetric zero biases
    x = ch.sample_features(64, d, 0.3, rng)
    xi = rng.standard_normal((64, n))
    mon = (rng.normal(0, 1, (d, n)), rng.normal(0, 0.1, d)) if mu > 0 else None
    _, _, g = ch.loss_and_grads(P, x, xi, sigma, lam=lam, monitor=mon, mu=mu)
    worst = 0.0
    for _ in range(n_checks):
        k = ch.PARAMS[rng.integers(len(ch.PARAMS))]
        idx = tuple(rng.integers(s) for s in P[k].shape)
        old = P[k][idx]
        P[k][idx] = old + eps; lp = ch.loss_and_grads(P, x, xi, sigma, lam, mon, mu)[0]
        P[k][idx] = old - eps; lm = ch.loss_and_grads(P, x, xi, sigma, lam, mon, mu)[0]
        P[k][idx] = old
        fd = (lp - lm) / (2 * eps)
        rel = abs(fd - g[k][idx]) / max(1e-8, abs(fd) + abs(g[k][idx]))
        worst = max(worst, rel)
    return worst


def test_all():
    for act in ("relu", "softplus"):
        ch.ACT = act
        for lam, mu in [(0, 0), (0.3, 0), (0, 0.7), (0.3, 0.7)]:
            w = check(lam, mu)
            print(f"act={act} lam={lam} mu={mu}: worst relative error {w:.2e}")
            assert w < 1e-5, w
    ch.ACT = "relu"


if __name__ == "__main__":
    test_all()
    print("gradient check passed")
