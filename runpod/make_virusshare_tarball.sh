#!/usr/bin/env bash
# Bundle the deduped VirusShare opcode corpus for one-time upload to the pod.
# Kept SEPARATE from the code tarball (make_upload_tarball.sh) so code re-uploads
# stay tiny — the dataset only needs to go up once.
# Produces ../virusshare.tar.gz with paths REPO/virusshare/...
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
REPO="$(basename "$PWD")"

find virusshare \( -name '.DS_Store' -o -name '._*' \) -delete 2>/dev/null || true

cd ..
COPYFILE_DISABLE=1 tar \
  --exclude='.DS_Store' --exclude='._*' \
  -czf "$REPO/../virusshare.tar.gz" \
  "$REPO/virusshare"

echo "Wrote $(cd "$REPO" && cd .. && pwd)/virusshare.tar.gz"
du -sh "$REPO/../virusshare.tar.gz"
echo "family count inside:"
tar -tzf "$REPO/../virusshare.tar.gz" | grep -oE "$REPO/virusshare/[^/]+/" | sort -u | wc -l
