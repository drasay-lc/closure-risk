#!/usr/bin/env bash
# Download the IPEDS complete data files used by this project (2001-2023 collections):
#   HD (directory), EFyyyyA (fall enrollment), EAP (employees by assignment),
#   Fyyyy_F2 (private nonprofit / FASB finance), Fyyyy_F3 (for-profit finance).
#
# Source: the public GitHub mirror of NCES "Complete Data Files" at
#   https://github.com/pbradl42/IPEDSpublic  (mirrored Feb 2025, through the 2023-24 collection)
# The same files can be downloaded directly from NCES:
#   https://nces.ed.gov/ipeds/datacenter/data/<FILE>.zip
# Set SOURCE=nces to pull from NCES instead of the mirror.
#
# Output: data/raw/<collection year>/<file>.csv   (lower-cased file names)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
MIRROR="$ROOT/data/mirror"
FIRST=${FIRST:-2001}
LAST=${LAST:-2023}
SOURCE=${SOURCE:-mirror}
mkdir -p "$RAW"

finance_tag() { # collection year -> FASB/for-profit finance file stem for that fiscal year
  local y=$1; printf "F%02d%02d" $(( (y-1) % 100 )) $(( y % 100 ))
}

files_for_year() {
  local y=$1 f; f=$(finance_tag "$y")
  [ "$y" -ge 2002 ] && echo "HD${y}"
  echo "EF${y}A" "EAP${y}" "${f}_F2" "${f}_F3"
}

if [ "$SOURCE" = "nces" ]; then
  for y in $(seq "$FIRST" "$LAST"); do
    mkdir -p "$RAW/$y"
    for s in $(files_for_year "$y"); do
      curl -sSfL "https://nces.ed.gov/ipeds/datacenter/data/${s}.zip" -o "/tmp/${s}.zip"
      unzip -o -q "/tmp/${s}.zip" -d "$RAW/$y"
    done
  done
else
  if [ ! -d "$MIRROR/.git" ]; then
    git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/pbradl42/IPEDSpublic "$MIRROR"
  fi
  : > /tmp/ipeds_paths.txt
  for y in $(seq "$FIRST" "$LAST"); do
    for s in $(files_for_year "$y"); do echo "/raw_data/$y/${s}.zip" >> /tmp/ipeds_paths.txt; done
  done
  git -C "$MIRROR" sparse-checkout set --no-cone --stdin < /tmp/ipeds_paths.txt
  git -C "$MIRROR" checkout -q HEAD
  for y in $(seq "$FIRST" "$LAST"); do
    mkdir -p "$RAW/$y"
    for z in "$MIRROR/raw_data/$y"/*.zip; do [ -f "$z" ] && unzip -o -q "$z" -d "$RAW/$y"; done
  done
fi
# normalise file names to lower case
find "$RAW" -name '*.csv' | while read -r f; do l="$(dirname "$f")/$(basename "$f" | tr 'A-Z' 'a-z')"; if [ "$f" != "$l" ]; then mv "$f" "$l"; fi; done
echo "IPEDS files in $RAW"
