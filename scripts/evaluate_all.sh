#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
uv run evaluate submissions/v57_soft_overlap_sa.py --all --json-out results/v57_eval.json \
  2>&1 | tee results/v57_eval.log
