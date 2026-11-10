"""E11 (supplement). How are features packed into coordinates in E11's starting codes?

Re-creates the legible codes that E11 starts from (same seeds, ReLU receiver) and records, for each,
how many public coordinates are used, how many are shared by two or more features, and the
largest number of features sharing one coordinate.
Output: results/e11_packing.json
"""
import json
import numpy as np
from bhi import channel as ch, drift

ch.ACT = "relu"


def main(seeds=5):
    out = []
    for d in (12, 32):
        cfg = dict(drift.DEFAULT, d=d)
        for seed in range(seeds):
            rng = np.random.default_rng(2000 + 37 * d + seed)      # as in e11_optimizers
            P, _, owner = drift.make_legible(cfg, rng)
            c = np.bincount(owner, minlength=cfg["n"])
            out.append(dict(d=d, seed=seed, used=int(np.sum(c > 0)), shared=int(np.sum(c >= 2)),
                            max_owners=int(c.max()), owners_per_coord=sorted(c.tolist(), reverse=True)))
            print(out[-1], flush=True)
    json.dump(out, open("results/e11_packing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
