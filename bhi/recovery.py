"""l1 recovery (basis pursuit) and the Donoho-Tanner / ALMT phase-transition curve."""
import numpy as np
from scipy.optimize import linprog, brentq
from scipy.stats import norm


def basis_pursuit(W, y):
    """argmin ||z||_1 subject to W z = y, as an LP in (u, v) >= 0 with z = u - v."""
    n, m = W.shape
    c = np.ones(2 * m)
    Aeq = np.hstack([W, -W])
    res = linprog(c, A_eq=Aeq, b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    return res.x[:m] - res.x[m:]


def nonneg_pursuit(W, y):
    """argmin sum(z) subject to W z = y, z >= 0 (the right decoder for nonnegative features)."""
    n, m = W.shape
    res = linprog(np.ones(m), A_eq=W, b_eq=y, bounds=(0, None), method="highs")
    return None if res.status != 0 else res.x


def recovery_rate(W, k, trials, rng, tol=1e-4, nonneg=False):
    """Fraction of random k-sparse vectors recovered exactly by l1 minimisation.
    Signed coefficients by default; nonneg=True uses nonnegative coefficients and decoder."""
    n, m = W.shape
    ok = 0
    for _ in range(trials):
        x = np.zeros(m)
        S = rng.choice(m, k, replace=False)
        mag = rng.uniform(0.5, 1.5, k)
        x[S] = mag if nonneg else rng.choice([-1.0, 1.0], k) * mag
        z = (nonneg_pursuit if nonneg else basis_pursuit)(W, W @ x)
        if z is not None and np.max(np.abs(z - x)) < tol * np.max(np.abs(x)):
            ok += 1
    return ok / trials


def _psi_tau(rho, tau):
    tail = (1 + tau ** 2) * norm.cdf(-tau) - tau * norm.pdf(tau)
    return rho * (1 + tau ** 2) + (1 - rho) * 2 * tail


def psi(rho):
    """Upper bound (tight up to O(1/sqrt m)) on delta(D(||.||_1, x))/m for k/m = rho."""
    taus = np.linspace(0, 8, 4001)
    return float(np.min(_psi_tau(rho, taus)))


def _psi_plus_tau(rho, tau):
    tail = (1 + tau ** 2) * norm.cdf(-tau) - tau * norm.pdf(tau)
    return rho * (1 + tau ** 2) + (1 - rho) * tail


def psi_plus(rho):
    """Nonnegative version: descent cone of sum(z) + indicator(z >= 0); only the upper
    tail of the off-support coordinates contributes."""
    taus = np.linspace(-4, 8, 6001)
    return float(np.min(_psi_plus_tau(rho, taus)))


def dt_threshold_k(n, m, nonneg=False):
    """Sparsity k at which l1 recovery transitions for a random n x m code: m psi(k/m) = n."""
    ps = psi_plus if nonneg else psi
    f = lambda k: m * ps(k / m) - n
    return brentq(f, 1e-6, m * 0.999)


def onestep_threshold_decode(W, x, thr=0.5):
    """ReLU-style one-step decoder for binary features: round(W^T W x > thr)."""
    return (W.T @ (W @ x) > thr).astype(float)
