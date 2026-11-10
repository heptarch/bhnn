#!/usr/bin/env bash
# Download the two embedding sources used in E03 (GitHub release assets) and convert them to .npz.
# Each download goes in its own directory; extraction scripts run with python -I from the repo.
set -euo pipefail
ROOT="${BHI_DATA:-/home/claude/data}"
mkdir -p "$ROOT/spacy_trf" "$ROOT/glove"
curl -sSL -o "$ROOT/spacy_trf/en_core_web_trf.whl" \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
curl -sSL -o "$ROOT/glove/glove100.gz" \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
python -I scripts/extract_roberta_embeddings.py "$ROOT/spacy_trf/en_core_web_trf.whl" "$ROOT/spacy_trf/roberta_embeddings.npz"
python -I scripts/extract_glove.py "$ROOT/glove/glove100.gz" "$ROOT/glove/glove100.npz"
