"""Sender -> noisy channel -> receiver: the toy model of Part II.

    x in {0,1}^d, i.i.d. Bernoulli(p)                 (feature vector)
    h  = tanh(A x + a)            in R^n              (message; the "public frame" is
                                                       the coordinate basis of R^n)
    y  = h + sigma * xi,  xi ~ N(0, I)                (channel noise)
    v  = B y + b,  u = act(v),  logits = C u + c      (receiver; act = relu, or softplus for Hessians)

Loss per example = sum over features of binary cross-entropy (mean over the batch).
Optional terms:
    lam * sum_j |h_j|                                 (l1 legibility budget in the public frame)
    mu  * BCE(sigmoid(M h + m0), x)                   (frozen linear monitor in the training loss)

Gradients are written out by hand; tests/test_gradients.py checks them against
central finite differences.
"""
import numpy as np

PARAMS = ("A", "a", "B", "b", "C", "c")
ACT = "relu"        # receiver hidden activation: "relu" (default) or "softplus" (smooth; used for Hessians)


def _act(v):
    return np.maximum(v, 0.0) if ACT == "relu" else np.logaddexp(0, v)


def _dact(v):
    return (v > 0).astype(float) if ACT == "relu" else 0.5 * (1.0 + np.tanh(0.5 * v))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def bce(logits, x):
    # numerically stable binary cross-entropy, summed over features, mean over batch
    return np.mean(np.sum(np.logaddexp(0, logits) - x * logits, axis=1))


def init_params(d, n, H, rng):
    return dict(
        A=rng.normal(0, 1 / np.sqrt(d), (n, d)), a=np.zeros(n),
        B=rng.normal(0, 1 / np.sqrt(n), (H, n)), b=np.zeros(H),
        C=rng.normal(0, 1 / np.sqrt(H), (d, H)), c=np.zeros(d),
    )


def sample_features(N, d, p, rng):
    return (rng.random((N, d)) < p).astype(float)


def forward(P, x, xi, sigma):
    z = x @ P["A"].T + P["a"]
    h = np.tanh(z)
    y = h + sigma * xi
    v = y @ P["B"].T + P["b"]
    u = _act(v)
    logits = u @ P["C"].T + P["c"]
    return dict(z=z, h=h, y=y, v=v, u=u, logits=logits)


def loss_and_grads(P, x, xi, sigma, lam=0.0, monitor=None, mu=0.0):
    """Return (total loss, task loss, grads dict) for a batch with given channel noise xi."""
    N = x.shape[0]
    f = forward(P, x, xi, sigma)
    task = bce(f["logits"], x)
    total = task
    dlog = (sigmoid(f["logits"]) - x) / N
    g = {}
    g["C"] = dlog.T @ f["u"]
    g["c"] = dlog.sum(0)
    du = dlog @ P["C"]
    dv = du * _dact(f["v"])
    g["B"] = dv.T @ f["y"]
    g["b"] = dv.sum(0)
    dh = dv @ P["B"]
    if lam > 0:
        total = total + lam * np.mean(np.sum(np.abs(f["h"]), axis=1))
        dh = dh + lam * np.sign(f["h"]) / N
    if monitor is not None and mu > 0:
        M, m0 = monitor
        ml = f["h"] @ M.T + m0
        total = total + mu * bce(ml, x)
        dh = dh + mu * ((sigmoid(ml) - x) / N) @ M
    dz = dh * (1 - f["h"] ** 2)
    g["A"] = dz.T @ x
    g["a"] = dz.sum(0)
    return total, task, g


def task_f1(P, x, xi, sigma):
    f = forward(P, x, xi, sigma)
    return f1_score((f["logits"] > 0).astype(float), x)


def f1_score(pred, x):
    tp = float((pred * x).sum())
    fp = float((pred * (1 - x)).sum())
    fn = float(((1 - pred) * x).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-12)


# ------------------------------------------------------- flat-vector utilities
def flatten(P):
    return np.concatenate([P[k].ravel() for k in PARAMS])


def unflatten(theta, like):
    out, i = {}, 0
    for k in PARAMS:
        s = like[k].size
        out[k] = theta[i:i + s].reshape(like[k].shape)
        i += s
    return out


def offsets(like):
    out, i = {}, 0
    for k in PARAMS:
        out[k] = i
        i += like[k].size
    return out
