#!/usr/bin/env bash
# Extract + content-dedup the 22 VirusShare families from the opcode zip into
# virusshare/<family>/VirusShare_<hash>.asm.txt  (matches malicia/ shape).
# Per-family: extract only that family from v077_clean, keep one file per unique
# content hash, then discard the temp dir so peak disk stays ~one family.
set -euo pipefail

ZIP="VirusShare_opcodes_extracted_v077.zip"
OUT="virusshare"
TMP=".vs_tmp"

FAMS=(
  Lamechi.B Diplugem Systex.A Startpage Delf Enterak.A DelfInject Small
  CeeInject Renos "Toga!rfn" Hotbar Injector FakeRean Expiro.BK OnLineGames
  Allaple.A VBInject Beebone Obfuscator Vundo Vobfus
)

mkdir -p "$OUT"
printf "%-14s %10s %10s\n" "family" "extracted" "unique"
printf "%-14s %10s %10s\n" "------" "---------" "------"

gtot=0; guniq=0
for fam in "${FAMS[@]}"; do
  rm -rf "$TMP"; mkdir -p "$TMP"
  unzip -q "$ZIP" "v077_clean/$fam/*" -d "$TMP"
  src="$TMP/v077_clean/$fam"
  dst="$OUT/$fam"; mkdir -p "$dst"

  total=$(find "$src" -name '*.txt' | wc -l | tr -d ' ')
  # md5 -r => "<hash> <path>"; keep first path per unique hash (deterministic via sort)
  kept=0
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    base=$(basename "$path" .txt)
    cp "$path" "$dst/$base.asm.txt"
    kept=$((kept+1))
  done < <(md5 -r "$src"/*.txt | sort | awk '{h=$1; $1=""; sub(/^[ \t]+/,""); if(!seen[h]++) print}')

  rm -rf "$TMP"
  printf "%-14s %10d %10d\n" "$fam" "$total" "$kept"
  gtot=$((gtot+total)); guniq=$((guniq+kept))
done

printf "%-14s %10d %10d\n" "TOTAL" "$gtot" "$guniq"
echo "Done -> $OUT/  (deduped, *.asm.txt)"
