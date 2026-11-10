"""E15. In superposition, is the legible code still a critical point? (Part II)

Starting from exactly the legible codes that E11 starts from (same seeds, ReLU receiver): at d = 12
features in n = 16 coordinates at most one coordinate is shared, at d = 32 nearly every coordinate is
shared by 2-4 features (E11 packing supplement).  Restrict the sender to each feature's argmax coordinate
(shared coordinates allowed), polish all parameters to a critical point *within* that
support (full batch), then compare the gradient norm within the support
with the gradient in leak directions (sender entries outside the support), separately for leaks into
coordinates owned by other features and into empty coordinates.  d = 12 is the control.
Output: results/e15_superposition_forces.json
"""
import json, time
import numpy as np
from bhi import channel as ch, drift
from experiments.e13_saddle_hessian import polish

ch.ACT = "relu"          # as in E11; only gradients are needed here, and ReLU is differentiable a.e.


def main(seeds=3, N=6144):
    out = dict(runs=[])
    t0 = time.time()
    for d in (12, 32):
        for seed in range(seeds):
            cfg = dict(drift.DEFAULT, d=d)
            rng = np.random.default_rng(2000 + 37 * d + seed)       # E11's seeds: these are E11's starting codes
            P, _, owner = drift.make_legible(cfg, rng)
            mask = np.zeros_like(P["A"]); mask[owner, np.arange(d)] = 1.0
            P["A"] *= mask
            X = ch.sample_features(N, d, cfg["p"], rng)
            XI = rng.standard_normal((N, cfg["n"]))
            polish(P, mask, X, XI, cfg["sigma"])
            _, _, g = ch.loss_and_grads(P, X, XI, cfg["sigma"])
            within = np.sqrt(sum(np.sum(v ** 2) for k, v in g.items() if k != "A") + np.sum((g["A"] * mask) ** 2))
            occ = np.zeros(cfg["n"], bool); occ[owner] = True
            owners_per_coord = np.bincount(owner, minlength=cfg["n"])
            leak_occ = (g["A"] * (1 - mask))[occ]          # leaks into coordinates owned by other features
            leak_emp = g["A"][~occ]                        # leaks into empty coordinates (zero by the lemma)
            r = dict(d=d, seed=seed, grad_within=float(within),
                     grad_leak_occupied=float(np.linalg.norm(leak_occ)),
                     grad_leak_occupied_rms=float(np.sqrt(np.mean(leak_occ ** 2))),
                     grad_leak_occupied_max=float(np.abs(leak_occ).max()),
                     grad_leak_empty=float(np.linalg.norm(leak_emp)) if leak_emp.size else 0.0,
                     n_empty=int((~occ).sum()), n_shared=int((owners_per_coord >= 2).sum()),
                     max_owners=int(owners_per_coord.max()), features_per_used_coord=float(d / occ.sum()),
                     read_empty=float(np.linalg.norm(P["B"][:, ~occ])) if (~occ).any() else 0.0,
                     read_used=float(np.linalg.norm(P["B"][:, occ])))
            np.savez(f"results/e15_params_d{d}_s{seed}.npz", owner=owner, **P)
            out["runs"].append(r)
            print(f"d={d} seed={seed}: |g| within {within:.1e}, leak-occupied {r['grad_leak_occupied']:.1e} "
                  f"(rms {r['grad_leak_occupied_rms']:.1e}, max {r['grad_leak_occupied_max']:.1e}), "
                  f"leak-empty {r['grad_leak_empty']:.1e}, empty {r['n_empty']}, shared {r['n_shared']} (max {r['max_owners']}), "
                  f"|B| empty {r['read_empty']:.2f} vs used {r['read_used']:.2f} [{time.time()-t0:.0f}s]",
                  flush=True)
    json.dump(out, open("results/e15_superposition_forces.json", "w"), indent=1)


if __name__ == "__main__":
    main()
