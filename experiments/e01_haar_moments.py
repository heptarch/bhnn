"""E01. Monte Carlo check of the exact second-moment formulas (Part I).

  complex Haar code     Var <psi|V^dag V|phi> = (M-N)(M-|c|^2) / (N (M^2-1))
  real Haar frame       Var <Wx,Wy>           = (m-n)(m+(m-2)c^2) / (n (m-1)(m+2))
  Gaussian (1/n)        Var <Wx,Wy>           = (1+c^2)/n
  Lubkin                E tr rho_A^2          = (dA+dB)/(dA dB + 1)
Output: results/e01_haar_moments.json
"""
import json
import numpy as np
from bhi import frames as fr


def unit(v):
    return v / np.linalg.norm(v)


def main(draws=3000):
    rng = np.random.default_rng(11)
    rows = []
    for M, N in [(64, 4), (64, 16), (64, 48)]:
        for target_c in (0.0, 0.6, 0.95):
            # complex pair with prescribed |<psi|phi>|
            psi = unit(rng.standard_normal(M) + 1j * rng.standard_normal(M))
            perp = rng.standard_normal(M) + 1j * rng.standard_normal(M)
            perp = unit(perp - np.vdot(psi, perp) * psi)
            phi = target_c * psi + np.sqrt(1 - target_c ** 2) * perp
            c = np.vdot(psi, phi)
            # real pair
            x = unit(rng.standard_normal(M))
            yp = rng.standard_normal(M); yp = unit(yp - (x @ yp) * x)
            y = target_c * x + np.sqrt(1 - target_c ** 2) * yp
            cv, rv, gv = [], [], []
            for _ in range(draws):
                V = fr.haar_code_complex(N, M, rng)
                cv.append(np.vdot(V @ psi, V @ phi))
                W = fr.haar_frame_real(N, M, rng)
                rv.append((W @ x) @ (W @ y))
                G = fr.gaussian_frame(N, M, rng)
                gv.append((G @ x) @ (G @ y))
            cv, rv, gv = map(np.array, (cv, rv, gv))
            rows.append(dict(
                M=M, N=N, c=target_c,
                complex_mean=float(abs(cv.mean() - c)), complex_var=float(np.mean(abs(cv - c) ** 2)),
                complex_pred=float(fr.var_overlap_complex(M, N, abs(c) ** 2)),
                real_var=float(np.var(rv)), real_pred=float(fr.var_overlap_real_haar(M, N, target_c)),
                gauss_var=float(np.var(gv)), gauss_pred=float(fr.var_overlap_gaussian(N, target_c)),
                real_kurt=float(np.mean((rv - rv.mean()) ** 4) / np.var(rv) ** 2),
                complex_abs_kurt=float(np.mean(abs(cv - c) ** 4) / np.mean(abs(cv - c) ** 2) ** 2),
            ))
            r = rows[-1]
            print(f"M={M} N={N} c={target_c}: complex {r['complex_var']:.5f}/{r['complex_pred']:.5f}  "
                  f"real {r['real_var']:.5f}/{r['real_pred']:.5f}  gauss {r['gauss_var']:.5f}/{r['gauss_pred']:.5f}  "
                  f"kurt real {r['real_kurt']:.2f} complex {r['complex_abs_kurt']:.2f}", flush=True)
    lub = []
    for dA in (2, 4, 8, 16, 32, 64, 128):
        dB = 16
        vals = []
        for _ in range(400):
            v = rng.standard_normal(dA * dB) + 1j * rng.standard_normal(dA * dB)
            v = unit(v).reshape(dA, dB)
            rho = v @ v.conj().T
            vals.append(np.trace(rho @ rho).real)
        lub.append(dict(dA=dA, dB=dB, mc=float(np.mean(vals)), pred=float(fr.lubkin_purity(dA, dB))))
        print(f"Lubkin dA={dA}: {lub[-1]['mc']:.4f} vs {lub[-1]['pred']:.4f}")
    json.dump(dict(draws=draws, overlaps=rows, lubkin=lub), open("results/e01_haar_moments.json", "w"), indent=1)


if __name__ == "__main__":
    main()
