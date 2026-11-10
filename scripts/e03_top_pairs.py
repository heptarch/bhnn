"""E03 (supplement): which vocabulary items have the largest overlaps?

For a random sample of 4000 items from each embedding matrix, list the most coherent pairs after
centring and after whitening, with their strings. Standalone (numpy + msgpack only) so that it
can be run isolated, since it reads the downloaded wheel:

    python -I scripts/e03_top_pairs.py [DATA_ROOT]

Output: results/e03_top_pairs.json (written relative to the current directory).
"""
import json, os, sys, zipfile
import numpy as np
import msgpack

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BHI_DATA", "/home/claude/data")


def roberta():
    E = np.load(f"{ROOT}/spacy_trf/roberta_embeddings.npz")[
        "curated_encoder__embeddings__inner__word_embeddings__weight"].astype(np.float64)
    w = zipfile.ZipFile(f"{ROOT}/spacy_trf/en_core_web_trf.whl")
    name = [n for n in w.namelist() if n.endswith("transformer/model")][0]
    obj = msgpack.unpackb(w.read(name), raw=False, strict_map_key=False)
    bp = msgpack.unpackb(obj["attrs"][3]["byte_bpe_processor"], raw=False, strict_map_key=False)
    inv = {v: k for k, v in bp["vocab"].items()}
    words = np.array([inv.get(i, "?") for i in range(len(E))], dtype=object)
    return E[4:], words[4:]                      # drop <s>, <pad>, </s>, <unk>, as in E03


def glove():
    z = np.load(f"{ROOT}/glove/glove100.npz", allow_pickle=False)
    return z["E"].astype(np.float64), z["words"].astype(object)


def versions(E):
    C = E - E.mean(0)
    ev, V = np.linalg.eigh(C.T @ C / len(C))
    return {"centred": C, "whitened": C @ (V @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-12))) @ V.T)}


def main(sample=4000, top=12):
    out = {}
    for name, load in (("roberta", roberta), ("glove", glove)):
        E, words = load()
        rng = np.random.default_rng(7)
        idx = rng.choice(len(E), sample, replace=False)
        out[name] = {}
        for v, X in versions(E).items():
            U = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
            G = np.abs(U @ U.T); np.fill_diagonal(G, 0)
            iu = np.triu_indices(sample, 1)
            order = np.argsort(-G[iu])[:top]
            pairs = [(str(words[idx[iu[0][o]]]), str(words[idx[iu[1][o]]]), float(G[iu][o])) for o in order]
            out[name][v] = pairs
            print(name, v, [(a, b, round(c, 2)) for a, b, c in pairs], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e03_top_pairs.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
