"""E02. Are trained superposition codes Haar-random?  (Part I)

Train Toy Models of Superposition (n = 32 neurons, m features, uniform importance) at
several sizes and sparsities, then compare the represented feature directions with
Haar frames of the same shape on:
  * second moment of overlaps (a frame identity fixes this for ANY tight frame),
  * excess kurtosis of overlaps (Haar ~ 0),
  * max coherence vs the Welch bound,
  * frame tightness (participation ratio of the frame operator / n; 1 = tight),
  * exact l1 recovery of k-sparse signed vectors vs the Donoho-Tanner prediction,
  * the TMS task loss itself (trained W vs a Haar frame with the best uniform bias).
Output: results/e02_tms_vs_haar.json
"""
import json, sys, time
import numpy as np
from bhi import frames as fr, recovery as rc, tms


def best_haar_loss(n, m, p, rng, reps=3):
    best = []
    for _ in range(reps):
        W = fr.unit_columns(fr.haar_frame_real(n, m, rng))
        losses = [tms.tms_loss(W, np.full(m, -bb), p, rng, N=4000) for bb in np.linspace(0, 0.6, 13)]
        best.append(min(losses))
    return float(np.mean(best))


def main(steps=8000):
    rng = np.random.default_rng(7)
    n = 32
    configs = [(64, 0.10), (64, 0.03), (128, 0.03), (128, 0.01), (256, 0.01), (256, 0.003)]
    out = dict(n=n, steps=steps, models=[])
    saved = {}
    t0 = time.time()
    for m, p in configs:
        W, b = tms.train_tms(n, m, p, rng, steps=steps)
        norms = np.linalg.norm(W, axis=0)
        rep = norms > 0.5
        Wr = W[:, rep]
        mr = int(rep.sum())
        st = fr.frame_stats(Wr)
        haar = [fr.frame_stats(fr.haar_frame_real(n, mr, rng)) for _ in range(5)]
        ctrl = {k: float(np.mean([h[k] for h in haar])) for k in haar[0]}
        # l1 recovery on unit-normalised represented directions: signed and nonnegative
        U = fr.unit_columns(Wr)
        H = fr.unit_columns(fr.haar_frame_real(n, mr, rng))
        rec = {}
        for mode, nn in (("signed", False), ("nonneg", True)):
            kstar = rc.dt_threshold_k(n, mr, nonneg=nn)
            kgrid = sorted(set(max(1, int(round(f * kstar))) for f in (0.5, 0.75, 1.0, 1.25, 1.5)))
            rec[mode] = dict(dt_k=kstar,
                             trained={k: rc.recovery_rate(U, k, 30, rng, nonneg=nn) for k in kgrid},
                             haar={k: rc.recovery_rate(H, k, 30, rng, nonneg=nn) for k in kgrid})
        saved[f"W_m{m}_p{p}"] = W
        saved[f"b_m{m}_p{p}"] = b
        loss_tr = tms.tms_loss(W, b, p, rng)
        loss_h = best_haar_loss(n, m, p, rng)
        # keep overlaps for histograms
        ov_tr = fr.offdiag_overlaps(Wr).tolist() if mr > 1 else []
        ov_h = fr.offdiag_overlaps(fr.haar_frame_real(n, mr, rng)).tolist() if mr > 1 else []
        record = dict(m=m, p=p, represented=mr, norms=norms.tolist(), trained=st, haar=ctrl,
                   recovery=rec,
                   tms_loss_trained=loss_tr, tms_loss_haar_best_bias=loss_h,
                   overlaps_trained=ov_tr, overlaps_haar=ov_h)
        out["models"].append(record)
        print(f"m={m} p={p}: represented {mr}/{m}; msq {st['mean_sq_overlap']:.4f} (untf {st['untf_prediction']:.4f}, haar {ctrl['mean_sq_overlap']:.4f}); "
              f"kurt {st['excess_kurtosis']:.2f} (haar {ctrl['excess_kurtosis']:.2f}); "
              f"maxcoh {st['max_coherence']:.3f} (haar {ctrl['max_coherence']:.3f}, welch {st['welch_bound']:.3f}); "
              f"tight {st['frame_participation']:.3f}; rec {json.dumps(rec)}; "
              f"loss tr {loss_tr:.4f} haar {loss_h:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open("results/e02_tms_vs_haar.json", "w"))
    np.savez_compressed("results/e02_tms_weights.npz", **saved)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
