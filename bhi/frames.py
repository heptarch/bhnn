"""Random codes and the exact second-moment formulas used in Part I of the notes.

Conventions: a code maps an M-dimensional effective space to an N-dimensional
fundamental space (N < M).  Columns of the code matrix are the images of the
effective basis vectors ("feature directions").
"""
import numpy as np


# ---------------------------------------------------------------- Haar sampling
def haar_unitary(M, rng):
    """Haar-random M x M unitary (QR of a complex Ginibre matrix with phase fix)."""
    Z = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def haar_orthogonal(m, rng):
    """Haar-random m x m orthogonal matrix."""
    Q, R = np.linalg.qr(rng.standard_normal((m, m)))
    return Q * np.sign(np.diag(R))


def haar_code_complex(N, M, rng):
    """AEHPV reduced model: V = sqrt(M/N) * (first N rows of a Haar unitary). Shape (N, M)."""
    return np.sqrt(M / N) * haar_unitary(M, rng)[:N]


def haar_frame_real(n, m, rng):
    """Real analogue: W = sqrt(m/n) * (first n rows of a Haar orthogonal). Shape (n, m).
    W W^T = (m/n) I exactly (a tight frame); W^T W = (m/n) P with P a Haar rank-n projector."""
    return np.sqrt(m / n) * haar_orthogonal(m, rng)[:n]


def gaussian_frame(n, m, rng):
    """Entries i.i.d. N(0, 1/n). Shape (n, m)."""
    return rng.standard_normal((n, m)) / np.sqrt(n)


def unit_columns(W):
    return W / np.linalg.norm(W, axis=0, keepdims=True)


# ---------------------------------------------------------- exact moment formulas
def var_overlap_complex(M, N, c2):
    """Var <psi|V^dag V|phi> for the Haar code, unit psi, phi with |<psi|phi>|^2 = c2.
    From E[Q (x) Q] = a I + b F (Schur-Weyl)."""
    return (M - N) * (M - c2) / (N * (M ** 2 - 1))


def var_overlap_real_haar(m, n, c):
    """Var <Wx, Wy> for the real Haar frame, unit x, y with <x,y> = c.
    From E[P (x) P] = alpha I + beta F + beta Omega (Brauer)."""
    return (m - n) * (m + (m - 2) * c ** 2) / (n * (m - 1) * (m + 2))


def var_overlap_gaussian(n, c):
    """Var <Wx, Wy> for i.i.d. N(0,1/n) entries (Isserlis)."""
    return (1 + c ** 2) / n


def untf_mean_square_overlap(m, n):
    """Mean of <w_i,w_j>^2 over i != j for ANY unit-norm tight frame of m vectors in R^n."""
    return (m - n) / (n * (m - 1))


def welch_bound(m, n):
    """Lower bound on max_{i!=j} |<w_i,w_j>| for m unit vectors in R^n (m > n)."""
    return np.sqrt((m - n) / (n * (m - 1)))


def lubkin_purity(dA, dB):
    """E tr rho_A^2 for a Haar-random pure state on C^dA (x) C^dB."""
    return (dA + dB) / (dA * dB + 1)


# ------------------------------------------------------------ frame diagnostics
def offdiag_overlaps(W, max_cols=None, rng=None):
    """Off-diagonal cosine overlaps of the (normalised) columns of W (upper triangle)."""
    U = unit_columns(W)
    if max_cols is not None and U.shape[1] > max_cols:
        idx = rng.choice(U.shape[1], max_cols, replace=False)
        U = U[:, idx]
    G = U.T @ U
    iu = np.triu_indices(G.shape[0], 1)
    return G[iu]


def frame_stats(W, max_cols=None, rng=None):
    """Summary statistics comparing a code to a unit-norm tight frame / Haar frame."""
    n, m = W.shape
    if max_cols is not None and m > max_cols:
        idx = rng.choice(m, max_cols, replace=False)
        W = W[:, idx]
        m = max_cols
    U = unit_columns(W)
    G = U.T @ U
    iu = np.triu_indices(m, 1)
    o = G[iu]
    S = U @ U.T                              # frame operator of normalised columns
    ev = np.linalg.eigvalsh(S)
    return dict(
        n=n, m=m,
        mean_sq_overlap=float(np.mean(o ** 2)),
        untf_prediction=float(untf_mean_square_overlap(m, n)),
        excess_kurtosis=float(np.mean(o ** 4) / np.mean(o ** 2) ** 2 - 3.0),
        max_coherence=float(np.max(np.abs(o))),
        welch_bound=float(welch_bound(m, n)),
        frame_participation=float(ev.sum() ** 2 / (ev ** 2).sum() / n),  # 1 = tight frame
        frame_condition=float(ev.max() / max(ev.min(), 1e-12)),
    )
