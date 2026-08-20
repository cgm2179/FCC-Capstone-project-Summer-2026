#!/usr/bin/env bash
# Point this clone at repo-managed hooks (strips Cursor co-author trailers).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$root/misc/.githooks/"* 2>/dev/null || true
git -C "$root" config core.hooksPath misc/.githooks
git -C "$root" config user.name "cgm2179"
git -C "$root" config user.email "cgm2179@columbia.edu"
echo "hooksPath=misc/.githooks; user=cgm2179 <cgm2179@columbia.edu>"
