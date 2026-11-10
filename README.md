# Interpretability and the black hole interior — working notes and code

Working notes (`notes/notes.pdf`) and the numpy code behind every number and figure in them.

**Part I** makes precise an analogy between the black hole interior and superposition in neural
networks. The non-isometric code of Akers–Engelhardt–Harlow–Penington–Vardhan and the restricted
isometry property of a network's feature directions are two cases of one theorem about random
compressions acting on thin sets. It then tests the randomness assumption on trained models (toy models
of superposition, RoBERTa-base token embeddings, GloVe).

**Part II** studies *legibility*: whether training keeps a message written in activations readable in a
public frame. It uses a toy sender → noisy channel → receiver model with linear monitors. A legible code is a
critical point of the training loss when its coordinates are unshared. The notes test how the choice of
optimizer decides whether training leaves it, and how far a curvature theory of that point holds up.

## How this was made

The research direction is David Wakeham's. The code, the experiments and most of the prose were produced
by Claude (Anthropic; Claude Opus 5.5) in conversation with him, September–November 2026. The notes
carry a provenance section and a status tag on every claim (proved / cited / checked numerically /
heuristic / analogy). They also list the earlier claims that the experiments overturned.

## Quick start

```bash
pip install -r requirements.txt
python -m tests.test_gradients            # hand-written backprop vs finite differences
python -m experiments.e14_bilinear_handshake   # 1 minute; the saddle in isolation
```

Full pipeline (about 2 h on 2 cores):

```bash
bash scripts/fetch_weights.sh             # RoBERTa via spaCy's wheel, GloVe via gensim-data (GitHub releases)
python -m experiments.e01_haar_moments
python -m experiments.e02_tms_vs_haar
python -m experiments.e03_real_embeddings
python -I scripts/e03_top_pairs.py
python -m experiments.e10_drift_conditions
python -m experiments.e11_optimizers
python -m experiments.e11_packing
python -m experiments.e13_saddle_hessian
python -m experiments.e13_fresh
python -m experiments.e14_bilinear_handshake
python -m experiments.e15_superposition_forces
python scripts/make_figures.py && python scripts/make_tables.py
cd notes && pdflatex notes && pdflatex notes
```

## Layout

| path | contents |
|---|---|
| `bhi/frames.py` | Haar unitary/orthogonal codes, Gaussian frames, exact overlap-moment formulas, frame diagnostics |
| `bhi/recovery.py` | basis pursuit (signed and nonnegative) and the statistical-dimension / Donoho–Tanner thresholds |
| `bhi/tms.py` | toy models of superposition (ReLU-output model), trained with Adam |
| `bhi/channel.py` | sender/channel/receiver model with hand-written gradients (ReLU or softplus receiver) |
| `bhi/optim.py` | Adam, SGD and per-parameter hybrids (e.g. SGD on the sender, Adam on the receiver) |
| `bhi/probes.py` | frozen, recalibrated and refit linear monitors; legibility metrics |
| `bhi/drift.py` | shared machinery for the drift experiments |
| `experiments/` | E01–E03 (Part I) and E10–E15 (Part II; there is no E12), each writing `results/*.json` |
| `scripts/` | weight extraction (no PyTorch; whitelist unpickler), figure and table generators |
| `notes/` | LaTeX source; `figs/` and `tables/` are generated |

## Safety note on the weight extraction

The RoBERTa weights ship inside a PyTorch pickle. `scripts/extract_roberta_embeddings.py` reads it
with an unpickler that resolves only a short whitelist of tensor-rebuild functions, mapped to numpy, so
nothing in the file is executed. Run it with `python -I`, as the fetch script does.
