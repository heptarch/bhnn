"""E13. Is the legible code a saddle, and does its curvature track the readout error? (Part II)

Setup (d = 12 features < n = 16 coordinates, smooth softplus receiver so that Hessians are
well defined; finite differences through ReLU kinks are not):
  1. train a legible code (l1 budget), assign each feature its own coordinate by an optimal
     matching, then restrict the sender to that legible support
     (A[j,l] = 0 unless j = owner(l)) and polish all parameters to a critical point of the
     *task* loss within that subspace, full batch, on a fixed dataset with fixed noise;
  2. report the gradient in leak directions, separately for leaks into empty coordinates
     and into coordinates owned by another feature;
  3. global smallest Hessian eigenvalue (Lanczos on Hessian-vector products) and where its
     eigenvector lives;
  4. per feature l, the block {leak A[j,l]} + {read B[:,j]} for the least saturated empty coordinate j:
     smallest eigenvalue, cross-term norm ||C||, and the Schur ("envelope") curvature, compared
     with the I-MMSE prediction  -phi'(a_j)^2 mmse_l / sigma^2.
Loss: sum over features of BCE, mean over the batch.
Output: results/e13_saddle_hessian.json
"""
import json, time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.optimize import linear_sum_assignment
from bhi import channel as ch, drift, optim

ch.ACT = "softplus"


def polish(P, mask, X, XI, sigma, steps=(6000, 3000, 2000), lrs=(3e-3, 1e-3, 2e-4)):
    for n_steps, lr in zip(steps, lrs):
        opt = optim.adam(P, lr)
        for _ in range(n_steps):
            _, _, g = ch.loss_and_grads(P, X, XI, sigma)
            g["A"] = g["A"] * mask
            opt.step(g)
    return P


SIGMAS = (0.10, 0.20, 0.40, 0.60, 0.90)


def main(sigmas=SIGMAS, N=6144, out_path="results/e13_saddle_hessian.json"):
    out = dict(N=N, activation="softplus", runs=[])
    t0 = time.time()
    for sigma in sigmas:
        cfg = dict(drift.DEFAULT, d=12, sigma=sigma)
        d, n, H = cfg["d"], cfg["n"], cfg["H"]
        rng = np.random.default_rng(int(1e4 * sigma))
        P, _, _ = drift.make_legible(cfg, rng)
        # one coordinate per feature: optimal matching on |A| (argmax can assign two features to one coordinate)
        rows, cols = linear_sum_assignment(-np.abs(P["A"]).T)
        owner = np.empty(d, int); owner[rows] = cols
        P["A"][owner, np.arange(d)] = np.sign(P["A"][owner, np.arange(d)] + 1e-12) * np.maximum(
            np.abs(P["A"][owner, np.arange(d)]), 0.5)
        mask = np.zeros_like(P["A"]); mask[owner, np.arange(d)] = 1.0
        P["A"] = P["A"] * mask
        X = ch.sample_features(N, d, cfg["p"], rng)
        XI = rng.standard_normal((N, n))
        polish(P, mask, X, XI, sigma)
        L0, _, g = ch.loss_and_grads(P, X, XI, sigma)
        occupied = np.zeros(n, bool); occupied[owner] = True
        gA = g["A"]
        grad_in = float(np.sqrt(sum(np.sum(v ** 2) for k, v in g.items() if k != "A") + np.sum((gA * mask) ** 2)))
        grad_leak_empty = float(np.linalg.norm(gA[~occupied]))
        leak_occ = gA[occupied] * (1 - mask[occupied])
        grad_leak_occ = float(np.linalg.norm(leak_occ))

        theta0 = ch.flatten(P)
        grad = lambda th: ch.flatten(ch.loss_and_grads(ch.unflatten(th, P), X, XI, sigma)[2])

        def hvp(v, r=1e-4):
            return (grad(theta0 + r * v) - grad(theta0 - r * v)) / (2 * r)

        op = LinearOperator((theta0.size, theta0.size), matvec=hvp, dtype=float)
        lam_g, vec_g = eigsh(op, k=1, which="SA", tol=1e-7, maxiter=5000)
        V = ch.unflatten(vec_g[:, 0], P)
        comp = dict(
            leak_empty=float(np.sum(V["A"][~occupied] ** 2)),
            leak_occupied=float(np.sum((V["A"][occupied] * (1 - mask[occupied])) ** 2)),
            sender_support=float(np.sum((V["A"] * mask) ** 2) + np.sum(V["a"] ** 2)),
            receiver_read_empty=float(np.sum(V["B"][:, ~occupied] ** 2)),
        )
        comp["receiver_other"] = float(1 - sum(comp.values()))

        off = ch.offsets(P)
        empties = np.where(~occupied)[0]
        # analyse the least saturated empty coordinate: a leak into a coordinate whose bias saturates tanh
        # (phi'(a_j) ~ 0) has no effect at first or second order, which says nothing about the saddle
        phi2_empty = (1 - np.tanh(P["a"][empties]) ** 2) ** 2
        j = int(empties[np.argmax(phi2_empty)])
        np.savez_compressed(f"results/e13_params_sigma{sigma:.2f}.npz", owner=owner, X=X, XI=XI, **P)
        idx = [off["A"] + j * d + l for l in range(d)] + [off["B"] + u * n + j for u in range(H)]
        Hb = np.zeros((len(idx), len(idx)))
        for c_, i in enumerate(idx):
            e = np.zeros(theta0.size); e[i] = 1.0
            Hb[:, c_] = hvp(e)[idx]
        Hb = 0.5 * (Hb + Hb.T)
        Hrr = Hb[d:, d:]
        f = ch.forward(P, X, XI, sigma)
        q = ch.sigmoid(f["logits"])
        mmse = np.mean((X - q) ** 2, axis=0)
        phi2 = (1 - np.tanh(P["a"][j]) ** 2) ** 2
        feats = []
        for l in range(d):
            sel = [l] + list(range(d, len(idx)))
            B = Hb[np.ix_(sel, sel)]
            w, Vb = np.linalg.eigh(B)
            Cv = B[0, 1:]
            schur = B[0, 0] - Cv @ np.linalg.solve(Hrr + 1e-12 * np.eye(H), Cv)
            feats.append(dict(l=l, mmse=float(mmse[l]), lam_min=float(w[0]),
                              sender_share=float(Vb[0, 0] ** 2), C_norm=float(np.linalg.norm(Cv)),
                              H_ee=float(B[0, 0]), schur=float(schur),
                              immse_pred=float(-phi2 * mmse[l] / sigma ** 2)))
        run = dict(sigma=sigma, loss=float(L0), n_empty=int(len(empties)), empty_coord=j,
                   a_j=float(P["a"][j]), phi2_j=float(phi2), a_empty=P["a"][empties].tolist(),
                   phi2_empty=phi2_empty.tolist(), Hrr_eigs=np.linalg.eigvalsh(Hrr).tolist(),
                   grad_within=grad_in, grad_leak_empty=grad_leak_empty, grad_leak_occupied=grad_leak_occ,
                   lam_global=float(lam_g[0]), global_vec=comp, features=feats)
        out["runs"].append(run)
        lam_b = np.array([x["lam_min"] for x in feats]); mm = np.array([x["mmse"] for x in feats])
        sc = np.array([x["schur"] for x in feats]); pr = np.array([x["immse_pred"] for x in feats])
        print(f"sigma={sigma}: empties={len(empties)} |g|within={grad_in:.1e} |g|leak-empty={grad_leak_empty:.1e} "
              f"|g|leak-occupied={grad_leak_occ:.1e} lam_global={lam_g[0]:.2e} vec={ {k: round(v, 2) for k, v in comp.items()} } "
              f"block lam [{lam_b.min():.2e},{lam_b.max():.2e}] corr(log|lam|,log mmse)="
              f"{np.corrcoef(np.log(-lam_b + 1e-15), np.log(mm + 1e-15))[0, 1]:.2f} "
              f"median schur/immse={np.median(sc / pr):.2f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS)
    ap.add_argument("--out", default="results/e13_saddle_hessian.json")
    a = ap.parse_args()
    # each sigma is seeded independently (seed = 1e4 * sigma), so runs over disjoint sigma lists can be merged
    main(tuple(a.sigmas), out_path=a.out)
