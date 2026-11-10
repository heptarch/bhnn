"""E14. The handshake saddle in isolation (Part II, conceptual figure).

    L(eps, r) = c * eps * r + (h/2) r^2 + (q/4) (eps^2 + r^2)^2,   with h = 0.05 c, q = c

eps = sender leak into an empty public coordinate, r = receiver read of that coordinate.
At the origin the gradient vanishes; the Hessian [[0, c], [c, h]] has eigenvalue
lambda_- = (h - sqrt(h^2 + 4 c^2)) / 2 < 0.  Gradient descent escapes at rate eta*|lambda_-|,
so its escape time grows like log(1/init) / (eta |lambda_-|); Adam's per-coordinate
normalisation makes the escape time essentially independent of c.
Output: results/e14_bilinear_handshake.json
"""
import json
import numpy as np


def grad(z, c, h, q):
    e, r = z
    s = e * e + r * r
    return np.array([c * r + q * s * e, c * e + h * r + q * s * r])


def run(opt, c, h=0.05, q=1.0, init=(1e-4, -3e-5), lr=0.05, steps=40000, thresh=0.3):
    z = np.array(init, float)
    m = np.zeros(2); v = np.zeros(2)
    traj = [z.copy()]
    t_esc = None
    for t in range(1, steps + 1):
        g = grad(z, c, h, q)
        if opt == "gd":
            z = z - lr * g
        else:
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            z = z - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 10 == 0:
            traj.append(z.copy())
        if t_esc is None and np.linalg.norm(z) > thresh:
            t_esc = t
    return dict(traj=np.array(traj).tolist(), t_esc=t_esc)


def main():
    out = dict(curves={}, scaling=[])
    for c in (1.0, 0.1):
        out["curves"][str(c)] = {"gd": run("gd", c, q=c, lr=0.05), "adam": run("adam", c, q=c, lr=0.003)}
    for c in np.logspace(-2.5, 0, 11):
        h = 0.05 * c
        lam = (h - np.sqrt(h * h + 4 * c * c)) / 2
        gd = run("gd", c, h=h, q=c, lr=0.05, steps=400000)["t_esc"]
        ad = run("adam", c, h=h, q=c, lr=0.003, steps=400000)["t_esc"]
        out["scaling"].append(dict(c=float(c), lam=float(lam), t_gd=gd, t_adam=ad,
                                   t_gd_pred=float(np.log(0.3 / 1e-4) / (0.05 * abs(lam)))))
        print(f"c={c:.4f} lam={lam:.4f} GD {gd} (pred {out['scaling'][-1]['t_gd_pred']:.0f}) Adam {ad}")
    json.dump(out, open("results/e14_bilinear_handshake.json", "w"))


if __name__ == "__main__":
    main()
