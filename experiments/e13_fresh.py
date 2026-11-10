"""E13 (supplement): does the polished receiver generalize, or has it memorized the noise sample?

E13 polishes on a fixed full batch (6144 samples, fixed channel noise), so its critical point is that of an
empirical loss. This script reloads the polished parameters saved by E13 and compares, for each sigma,
the receiver's residual error on that batch with its residual error on fresh noise, and recomputes the
I-MMSE prediction of Prop. (envelope curvature) with the fresh-noise error.
Output: results/e13_fresh.json
"""
import json
import numpy as np
from scipy.stats import norm
from bhi import channel as ch

ch.ACT = "softplus"


def main(sigmas=(0.10, 0.20, 0.40, 0.60, 0.90), n_fresh=40000):
    e13 = {r["sigma"]: r for r in json.load(open("results/e13_saddle_hessian.json"))["runs"]}
    out = []
    for s in sigmas:
        z = np.load(f"results/e13_params_sigma{s:.2f}.npz")
        P = {k: z[k] for k in ch.PARAMS}
        owner, X, XI = z["owner"], z["X"], z["XI"]
        d, n = X.shape[1], XI.shape[1]
        A, a = P["A"][owner, np.arange(d)], P["a"][owner]
        gap = np.abs(np.tanh(A + a) - np.tanh(a))
        q = ch.sigmoid(ch.forward(P, X, XI, s)["logits"])
        rng = np.random.default_rng(1)
        Xf = ch.sample_features(n_fresh, d, 0.1, rng)
        XIf = rng.standard_normal((n_fresh, n))
        qf = ch.sigmoid(ch.forward(P, Xf, XIf, s)["logits"])
        mm_batch = np.mean((X - q) ** 2, axis=0)
        mm_fresh = np.mean((Xf - qf) ** 2, axis=0)
        r = e13[s]
        phi2 = r["phi2_j"]
        j = r["empty_coord"]
        read_empty = float(np.linalg.norm(P["B"][:, j]))
        read_owned = float(np.median(np.linalg.norm(P["B"][:, owner], axis=0)))
        out.append(dict(sigma=s, gap_median=float(np.median(gap)), gap_min=float(gap.min()),
                        mmse_batch=mm_batch.tolist(), mmse_fresh=mm_fresh.tolist(),
                        bit_error_fresh=float(np.mean((qf > 0.5) != Xf)),
                        midpoint_rule_error=float(np.mean(norm.sf(gap / (2 * s)))),
                        immse_pred_fresh=(-phi2 * mm_fresh / s ** 2).tolist(),
                        read_empty=read_empty, read_owned_median=read_owned,
                        envelope_positive=int(sum(x["schur"] > 0 for x in r["features"]))))
        print(f"sigma={s}: gap {np.median(gap):.2f}; mmse batch {mm_batch.mean():.1e}, fresh {mm_fresh.mean():.1e}; "
              f"I-MMSE (fresh) median {np.median(-phi2 * mm_fresh / s ** 2):.1e}; receiver weight on empty coord "
              f"{read_empty:.1f} vs owned {read_owned:.1f}; envelope > 0 for {out[-1]['envelope_positive']}/{d}", flush=True)
    json.dump(out, open("results/e13_fresh.json", "w"), indent=1)


if __name__ == "__main__":
    main()
