"""Toy Models of Superposition (Elhage et al. 2022), ReLU-output model:

    h = W x,   x_hat = relu(W^T h + b),   loss = mean_batch sum_i I_i (x_i - x_hat_i)^2

with x_i ~ Uniform[0,1] with probability p = 1 - S and 0 otherwise.  Trained with Adam.
"""
import numpy as np


def train_tms(n, m, p, rng, steps=10000, batch=1024, lr=1e-3, importance=None, log_every=0):
    I = np.ones(m) if importance is None else importance
    W = rng.normal(0, np.sqrt(2.0 / (n + m)), (n, m))      # Xavier-normal as in the TMS code
    b = np.zeros(m)
    mW, vW, mb, vb = (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
    for t in range(1, steps + 1):
        x = rng.random((batch, m)) * (rng.random((batch, m)) < p)
        h = x @ W.T
        pre = h @ W + b
        xh = np.maximum(pre, 0)
        r = (xh - x) * I                                    # dL/dxh / 2
        dpre = 2 * r * (pre > 0) / batch
        gb = dpre.sum(0)
        gW = h.T @ dpre + (dpre @ W.T).T @ x                # both uses of W
        for g, mm, vv, P in ((gW, mW, vW, W), (gb, mb, vb, b)):
            mm *= 0.9; mm += 0.1 * g
            vv *= 0.999; vv += 0.001 * g * g
            P -= lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        if log_every and t % log_every == 0:
            loss = np.mean(np.sum(I * (xh - x) ** 2, 1))
            print(f"  step {t} loss {loss:.4f}", flush=True)
    return W, b


def tms_loss(W, b, p, rng, N=20000, importance=None):
    m = W.shape[1]
    I = np.ones(m) if importance is None else importance
    x = rng.random((N, m)) * (rng.random((N, m)) < p)
    xh = np.maximum((x @ W.T) @ W + b, 0)
    return float(np.mean(np.sum(I * (xh - x) ** 2, 1)))
