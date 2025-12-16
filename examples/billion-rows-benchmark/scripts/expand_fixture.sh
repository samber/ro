#!/usr/bin/env bash
# Expand a small sample to N lines by repeating its contents. Usage:
#   ./expand_fixture.sh <sample> <out> <N>
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <sample> <out> <N>"
  exit 2
fi
sample=$1
out=$2
N=$3

if [ ! -f "$sample" ]; then
  echo "sample file not found: $sample"
  exit 2
fi

# Count lines in sample
lines=$(wc -l < "$sample" | tr -d ' ')
if [ "$lines" -eq 0 ]; then
  echo "sample file is empty"
  exit 2
fi

# Repeat sample lines until we reach N lines
> "$out"
while [ "$(wc -l < "$out" | tr -d ' ')" -lt "$N" ]; do
  cat "$sample" >> "$out"
done

# If we overshot, trim
if [ "$(wc -l < "$out" | tr -d ' ')" -gt "$N" ]; then
  head -n "$N" "$out" > "$out.tmp" && mv "$out.tmp" "$out"
fi

echo "wrote $out with $(wc -l < "$out" | tr -d ' ') lines"
