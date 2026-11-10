"""E03. Are the feature directions of real trained models Haar-random?  (Part I)

Codes tested ("features" = vocabulary items, "fundamental space" = embedding dimension):
  RoBERTa-base token embeddings   m = 50265 tokens in n = 768
  GloVe (wiki-gigaword, 100d)     m = 400000 words  in n = 100
Each in three versions:
  raw       rows normalised to unit length
  centred   mean embedding subtracted, then normalised
  whitened  centred, multiplied by Sigma^{-1/2} (forces a tight frame), then normalised
Compared with a Haar frame of the same (n, m_sample) on: n * mean square overlap,
excess kurtosis of overlaps, tail mass P(|cos| > t), max coherence of a sample,
frame tightness, and exact l1 recovery of k-sparse signed combinations.

Inputs (see scripts/): data/roberta_embeddings.npz, data/glove100.npz
Output: results/e03_real_embeddings.json
"""
import json, sys, time
import numpy as np
from bhi import frames as fr, recovery as rc

import os
ROOT = os.environ.get("BHI_DATA", "/home/claude/data")


def load(name):
    if name == "roberta":
        E = np.load(f"{ROOT}/spacy_trf/roberta_embeddings.npz")[
            "curated_encoder__embeddings__inner__word_embeddings__weight"]
        return E[4:].astype(np.float32)        # drop <s>, <pad>, </s>, <unk>
    return np.load(f"{ROOT}/glove/glove100.npz")["E"].astype(np.float32)


def versions(E):
    mu = E.mean(0, dtype=np.float64)
    C = (E - mu).astype(np.float32)
    cov = (C.T.astype(np.float64) @ C.astype(np.float64)) / len(C)
    ev, V = np.linalg.eigh(cov)
    Wh = (C @ (V @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-12))) @ V.T).astype(np.float32))
    out = {"raw": E, "centred": C, "whitened": Wh}
    info = dict(mean_norm_ratio=float(np.linalg.norm(mu) / np.mean(np.linalg.norm(E, axis=1))),
                cov_eigs=sorted((ev / ev.mean()).tolist(), reverse=True))
    return out, info


def overlap_stats(U, n, rng, n_pairs=400000, chunk=20000):
    parts = []
    for _ in range(n_pairs // chunk):
        i = rng.integers(0, len(U), chunk)
        j = rng.integers(0, len(U), chunk)
        keep = i != j
        parts.append(np.einsum("ij,ij->i", U[i[keep]], U[j[keep]]).astype(np.float64))
    o = np.concatenate(parts)
    m2 = np.mean(o ** 2)
    hist, edges = np.histogram(o * np.sqrt(n), bins=np.linspace(-8, 8, 161))
    tails = {str(t): float(np.mean(np.abs(o) > t)) for t in (0.2, 0.3, 0.5, 0.7)}
    return dict(n_mean_sq=float(n * m2), mean=float(np.mean(o)),
                excess_kurtosis=float(np.mean(o ** 4) / m2 ** 2 - 3), tails=tails,
                hist=hist.tolist(), edges=edges.tolist())


def main():
    rng = np.random.default_rng(3)
    out = {}
    t0 = time.time()
    for name in ("glove", "roberta"):
        E = load(name)
        n = E.shape[1]
        vers, info = versions(E)
        res = dict(n=n, m=len(E), **info, versions={})
        # Haar control overlaps: Gaussian-like with variance ~ 1/n for m >> n
        sample_m = 4000
        Hc = fr.unit_columns(fr.haar_frame_real(n, sample_m, rng)).T
        res["haar_overlaps"] = overlap_stats(Hc, n, rng)
        res["haar_frame"] = fr.frame_stats(Hc.T)
        for vname, X in vers.items():
            U = X / np.linalg.norm(X, axis=1, keepdims=True)
            idx = rng.choice(len(U), sample_m, replace=False)
            st = fr.frame_stats(U[idx].T.astype(np.float64))
            ov = overlap_stats(U, n, rng)
            res["versions"][vname] = dict(frame=st, overlaps=ov)
            print(f"{name} {vname}: n*E[cos^2]={ov['n_mean_sq']:.2f} mean={ov['mean']:.3f} "
                  f"kurt={ov['excess_kurtosis']:.2f} tails={ov['tails']} maxcoh(4000)={st['max_coherence']:.3f} "
                  f"(haar {res['haar_frame']['max_coherence']:.3f}) tight={st['frame_participation']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        # l1 recovery on a random sub-dictionary with m_sub = 2n (delta = 1/2)
        m_sub = 2 * n
        kstar = rc.dt_threshold_k(n, m_sub)
        fracs = (0.75, 1.0, 1.25) if n > 200 else (0.5, 0.75, 0.9, 1.0, 1.1, 1.25)
        ks = sorted(set(int(round(f * kstar)) for f in fracs))
        trials = 8 if n > 200 else 30
        rec = {}
        for vname in ("centred", "whitened"):
            X = vers[vname]
            idx = rng.choice(len(X), m_sub, replace=False)
            U = (X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)).T.astype(np.float64)
            rec[vname] = {k: rc.recovery_rate(U, k, trials, rng) for k in ks}
            print(f"  recovery {vname}: {rec[vname]} [{time.time()-t0:.0f}s]", flush=True)
        H = fr.unit_columns(fr.haar_frame_real(n, m_sub, rng))
        rec["haar"] = {k: rc.recovery_rate(H, k, trials, rng) for k in ks}
        print(f"  recovery haar: {rec['haar']}  DT k*={kstar:.1f}", flush=True)
        res["recovery"] = dict(m_sub=m_sub, dt_k=kstar, trials=trials, rates=rec)
        out[name] = res
        del E, vers
        json.dump(out, open("results/e03_real_embeddings.json", "w"))
    json.dump(out, open("results/e03_real_embeddings.json", "w"))


if __name__ == "__main__":
    main()
