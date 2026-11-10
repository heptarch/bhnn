"""Linear monitors and legibility metrics.

A monitor is a logistic probe x_hat = sigmoid(M h + m0) reading the message h.
"Frozen" = fit once at a legible checkpoint and never refit; "refit" = fit afresh on
the current code.  Frozen-minus-refit measures frame drift; receiver-minus-refit
measures information the receiver uses that no linear public decoder recovers.
"""
import numpy as np
from .channel import sigmoid, f1_score


def fit_probe(H, X, steps=1500, lr=1e-2, l2=1e-4):
    """Full-batch Adam on logistic loss, one output per feature."""
    N, n = H.shape
    d = X.shape[1]
    M, m0 = np.zeros((d, n)), np.zeros(d)
    mM, vM, mb, vb = np.zeros_like(M), np.zeros_like(M), np.zeros(d), np.zeros(d)
    for t in range(1, steps + 1):
        q = sigmoid(H @ M.T + m0)
        dl = (q - X) / N
        gM = dl.T @ H + l2 * M
        gb = dl.sum(0)
        mM = 0.9 * mM + 0.1 * gM; vM = 0.999 * vM + 0.001 * gM ** 2
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
        M -= lr * (mM / c1) / (np.sqrt(vM / c2) + 1e-8)
        m0 -= lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
    return M, m0


def fit_recalibration(probe, H, X, steps=600, lr=2e-2):
    """Keep the frozen probe's directions; refit only a per-feature scale and bias.
    Separates calibration drift (amplitudes/offsets) from directional drift."""
    M, m0 = probe
    S = H @ M.T                                    # frozen scores, one per feature
    N, d = S.shape
    al, be = np.ones(d), m0.copy()
    st = [np.zeros(d) for _ in range(4)]
    for t in range(1, steps + 1):
        q = sigmoid(al * S + be)
        dl = (q - X) / N
        ga, gb = (dl * S).sum(0), dl.sum(0)
        for i, (g, P) in enumerate(((ga, al), (gb, be))):
            st[2 * i] = 0.9 * st[2 * i] + 0.1 * g
            st[2 * i + 1] = 0.999 * st[2 * i + 1] + 0.001 * g * g
            P -= lr * (st[2 * i] / (1 - 0.9 ** t)) / (np.sqrt(st[2 * i + 1] / (1 - 0.999 ** t)) + 1e-8)
    return (M * al[:, None], m0 * 0 + be)


def column_hoyer(A):
    """Mean over features of the Hoyer sparsity of the sender column in the public frame
    (1 = each feature drives one neuron: monosemantic; 0 = spread evenly)."""
    n = A.shape[0]
    l1 = np.abs(A).sum(0)
    l2 = np.linalg.norm(A, axis=0) + 1e-12
    return float(np.mean((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)))


def neurons_per_feature(A):
    """Mean participation ratio of sender columns: effective number of neurons per feature."""
    a2 = A ** 2
    return float(np.mean(a2.sum(0) ** 2 / ((a2 ** 2).sum(0) + 1e-24)))


def probe_f1(probe, H, X):
    M, m0 = probe
    return f1_score(((H @ M.T + m0) > 0).astype(float), X)


def hoyer(H):
    """Hoyer sparsity in the public frame, averaged over messages: 1 = one-hot, 0 = flat."""
    n = H.shape[1]
    l1 = np.abs(H).sum(1)
    l2 = np.linalg.norm(H, axis=1) + 1e-12
    return float(np.mean((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)))


def active_coords(H, thr=0.3):
    """Mean number of public coordinates carrying signal (|h_j| > thr)."""
    return float((np.abs(H) > thr).sum(1).mean())


def participation_ratio(H):
    ev = np.linalg.eigvalsh(np.cov(H.T))
    return float(ev.sum() ** 2 / (ev ** 2).sum())
