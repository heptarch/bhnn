"""E11. Does the update rule decide whether a legible code stays legible? (Part II)

From a legible checkpoint, remove the l1 budget (no change of task) and continue with:
  adam            Adam everywhere, lr 3e-3
  sgd             plain SGD everywhere, lr 0.05
  sgd_large       plain SGD everywhere, lr 0.5
  hybrid          SGD (lr 0.05) on the sender/channel, Adam (3e-3) on the receiver
  hybrid_reverse  Adam (3e-3) on the sender/channel, SGD (0.05) on the receiver   [control]
Two feature counts: d=12 < n=16 (no superposition) and d=32 > n (superposition).
Seeds: 5.  Output: results/e11_optimizers.json
"""
import json, sys, time
import numpy as np
from bhi import drift, optim
from bhi.optim import PerParam, SENDER

OPTS = {
    "adam": lambda P: optim.adam(P, 3e-3),
    "sgd": lambda P: optim.sgd(P, 0.05),
    "sgd_large": lambda P: optim.sgd(P, 0.5),
    "hybrid": lambda P: optim.hybrid(P, sgd_lr=0.05, adam_lr=3e-3),
    "hybrid_reverse": lambda P: PerParam(P, {k: (("adam", 3e-3) if k in SENDER else ("sgd", 0.05))
                                             for k in P}),
}


def main(seeds=5, steps=6000):
    out = dict(steps=steps, seeds=seeds, runs=[])
    t0 = time.time()
    for d in (12, 32):
        cfg = dict(drift.DEFAULT, d=d)
        out.setdefault("configs", {})[str(d)] = cfg
        for seed in range(seeds):
            rng = np.random.default_rng(2000 + 37 * d + seed)
            P0, mon, owner = drift.make_legible(cfg, rng)
            for name, mk in OPTS.items():
                rows, _ = drift.run_condition(P0, mon, owner, cfg, rng, mk, steps)
                out["runs"].append(dict(d=d, seed=seed, optimizer=name, rows=rows))
                last = rows[-1]
                print(f"d={d} seed {seed} {name:15s} task {last['task']:.3f} frozen {last['frozen']:.3f} "
                      f"refit {last['refit']:.3f} hoyer {last['hoyer']:.2f} leak {last['leak']:.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open("results/e11_optimizers.json", "w"), indent=1)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
