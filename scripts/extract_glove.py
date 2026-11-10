"""Convert gensim-data's glove-wiki-gigaword-100 (word2vec text format, gzipped) to .npz.

Source: GitHub release RaRe-Technologies/gensim-data, tag glove-wiki-gigaword-100.
Usage:  python -I scripts/extract_glove.py IN.gz OUT.npz
"""
import gzip, sys
import numpy as np


def main(src, out):
    with gzip.open(src, "rt", encoding="utf8") as f:
        header = f.readline().split()
        n_words, dim = int(header[0]), int(header[1])
        E = np.empty((n_words, dim), dtype=np.float32)
        words = []
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            E[i] = np.asarray(parts[1:], dtype=np.float32)
    np.savez_compressed(out, E=E[: len(words)], words=np.array(words))
    print(E.shape, "saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
