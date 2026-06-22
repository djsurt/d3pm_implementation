#!/usr/bin/env bash
# Build a CLEAN tarball of the repo to upload to the RunPod pod.
# Run from anywhere:  bash runpod/make_upload_tarball.sh
# Produces ../d3pm_upload.tar.gz (one level above the repo, so it can't include
# itself). Excludes venvs, old runs, PDFs, the local-only comparison/ suite, and
# all macOS junk (.DS_Store / ._* AppleDouble) so none of it lands on the pod.
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
REPO="$(basename "$PWD")"              # e.g. d3pm_implementation
OUT="../d3pm_upload.tar.gz"

# Strip any macOS junk that already exists before packing.
find . -path ./.venv -prune -o -path ./comparison/.venv -prune -o \
  \( -name '.DS_Store' -o -name '._*' \) -print -delete 2>/dev/null || true

cd ..                                   # so paths in the tar are REPO/...
COPYFILE_DISABLE=1 tar \
  --exclude="$REPO/.venv" \
  --exclude="$REPO/comparison" \
  --exclude="$REPO/runs" \
  --exclude="$REPO/.claude" \
  --exclude="$REPO/.pytest_cache" \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='*.pdf' \
  -czf "$REPO/../d3pm_upload.tar.gz" \
  "$REPO/d3pm" "$REPO/tests" "$REPO/verify" "$REPO/runpod" "$REPO/malicia" \
  "$REPO/requirements.txt" "$REPO/CLAUDE.md" "$REPO/PLAN.md" "$REPO/FINDINGS.md"

echo "Wrote $(cd "$REPO" && cd .. && pwd)/d3pm_upload.tar.gz"
du -sh "$REPO/../d3pm_upload.tar.gz"
echo "Sanity — no junk inside:"
tar -tzf "$REPO/../d3pm_upload.tar.gz" | grep -E '\.DS_Store|\._|\.venv|/comparison/' | head || echo "  (clean)"
