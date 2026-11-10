"""E10. Legibility under four training conditions (Part II, Table 1 / Fig. drift).

From a legible checkpoint (l1 budget, p=0.10), continue for 6000 Adam steps under:
  release           : same task, l1 budget removed
  pressure          : feature density p 0.10 -> 0.30, budget removed
  pressure+budget   : p -> 0.30, budget kept
  pressure+monitor  : p -> 0.30, frozen monitor's loss added to the training loss
Seeds: 5.  Output: results/e10_drift_conditions.json
"""
import json, sys, time
import numpy as np
from bhi import drift, optim

CONDITIONS = {
    "release":          dict(p=0.10, lam=0.0, mu=0.0),
    "pressure":         dict(p=0.30, lam=0.0, mu=0.0),
    "pressure+budget":  dict(p=0.30, lam=1.0, mu=0.0),
    "pressure+monitor": dict(p=0.30, lam=0.0, mu=1.0),
}


def main(seeds=5, steps=6000):
    cfg = dict(drift.DEFAULT)
    out = dict(config=cfg, steps=steps, seeds=seeds, runs=[])
    t0 = time.time()
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        P0, mon, owner = drift.make_legible(cfg, rng)
        for name, c in CONDITIONS.items():
            rows, _ = drift.run_condition(P0, mon, owner, cfg, rng, lambda P: optim.adam(P, 3e-3),
                                          steps, p=c["p"], lam=c["lam"], mu=c["mu"])
            out["runs"].append(dict(seed=seed, condition=name, rows=rows))
            last = rows[-1]
            print(f"seed {seed} {name:17s} task {last['task']:.3f} frozen {last['frozen']:.3f} "
                  f"refit {last['refit']:.3f} hoyer {last['hoyer']:.2f} leak {last['leak']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open("results/e10_drift_conditions.json", "w"), indent=1)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
