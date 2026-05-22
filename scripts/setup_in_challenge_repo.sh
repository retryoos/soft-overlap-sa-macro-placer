#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/setup_in_challenge_repo.sh /path/to/macro-place-challenge-2026" >&2
  exit 2
fi

challenge_repo="$1"
if [[ ! -d "$challenge_repo/submissions" || ! -d "$challenge_repo/macro_place" ]]; then
  echo "error: expected a Partcl/HRT macro-place-challenge-2026 checkout" >&2
  exit 2
fi

mkdir -p "$challenge_repo/submissions/retryoos"
cp placer.py "$challenge_repo/submissions/soft_overlap_sa_macro_placer.py"
cp submissions/__init__.py "$challenge_repo/submissions/__init__.py"
cp submissions/retryoos/__init__.py "$challenge_repo/submissions/retryoos/__init__.py"
cp submissions/retryoos/incremental_sa.py "$challenge_repo/submissions/retryoos/incremental_sa.py"
cp submissions/retryoos/soft_overlap_sa.py "$challenge_repo/submissions/retryoos/soft_overlap_sa.py"
cp submissions/retryoos/replace_sa.py "$challenge_repo/submissions/retryoos/replace_sa.py"

echo "Installed Soft-Overlap SA placer into: $challenge_repo/submissions/soft_overlap_sa_macro_placer.py"
echo "Run: cd \"$challenge_repo\" && uv run evaluate submissions/soft_overlap_sa_macro_placer.py --all"
