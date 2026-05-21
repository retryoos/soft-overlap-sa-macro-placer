#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
uv run evaluate submissions/soft_overlap_sa_macro_placer.py --all \
  --json-out results/soft_overlap_sa_eval.json \
  2>&1 | tee results/soft_overlap_sa_eval.log
