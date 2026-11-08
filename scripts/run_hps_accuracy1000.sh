#!/bin/sh
# BusyBox-compatible. sh /mnt/usb/hps_accuracy1000/run.sh /home/root/hps_accuracy1000
set -eu
BUNDLE=$(CDPATH= cd "$(dirname "$0")" && pwd)
OUT=${1:-/home/root/hps_accuracy1000}
case "$OUT" in /*) ;; *) echo 'Use an absolute output path' >&2; exit 1;; esac
cd "$BUNDLE"
echo 'Checking USB bundle contents (this may take several minutes)...'
while read -r expected name; do
    actual=$(md5sum "$name")
    set -- $actual
    if [ "$1" != "$expected" ]; then
        echo "Checksum mismatch: $name" >&2
        exit 1
    fi
done < MD5SUMS
mkdir -p "$OUT"
identity=$(md5sum MD5SUMS)
if [ -f "$OUT/bundle.identity" ]; then
    [ "$(cat "$OUT/bundle.identity")" = "$identity" ] || { echo 'Output belongs to another bundle'; exit 1; }
else
    if [ -d "$OUT/results" ]; then echo 'Existing output has no bundle identity'; exit 1; fi
    printf '%s\n' "$identity" > "$OUT/bundle.identity"
fi
mkdir -p "$OUT/build" "$OUT/results" "$OUT/compact"
cp build/model_runner "$OUT/build/model_runner"
chmod +x "$OUT/build/model_runner"
cp experiment.json "$OUT/experiment.json"
cp SHA256SUMS "$OUT/SHA256SUMS"
cp MD5SUMS "$OUT/MD5SUMS"
{ uname -a; cat /proc/cpuinfo; cat /proc/meminfo; } > "$OUT/environment.txt"
for variant in fp32 int8; do
    precision=fp32
    [ "$variant" != int8 ] || precision=runtime-int8
    for list in "$BUNDLE"/inputs/batch_*.tsv; do
        batch=$(basename "$list" .tsv)
        dest="$OUT/results/${variant}_${batch}"
        if [ -f "$dest/COMPLETE" ]; then echo "Preserved completed $variant $batch"; continue; fi
        if [ -e "$dest" ]; then
            echo "Incomplete directory: $dest. Preserve/rename it before retrying." >&2
            exit 1
        fi
        echo "Running $variant $batch (100 images, 2 HPS threads)..."
        "$OUT/build/model_runner" eval "$BUNDLE/model/$variant.param" "$BUNDLE/model/$variant.bin" \
          "$list" "$dest" 2 0 100 in0 out0 0 "$precision" > "$dest.stdout" 2> "$dest.stderr"
        test -s "$dest/run.json"
        touch "$dest/COMPLETE"
    done
done
# Compact accuracy evidence: retain full logits in results on board.
for variant in fp32 int8; do
    for list in "$BUNDLE"/inputs/batch_*.tsv; do
        batch=$(basename "$list" .tsv)
        name="${variant}_${batch}"
        mkdir -p "$OUT/compact/$name"
        cp "$OUT/results/$name/run.json" "$OUT/compact/$name/"
        cp "$OUT/results/$name/image_ids.json" "$OUT/compact/$name/"
    done
done
cd "$OUT"
tar -czf accuracy1000_compact.tar.gz compact experiment.json environment.txt bundle.identity SHA256SUMS MD5SUMS
echo "Finished both 1000-image evaluations: $OUT/accuracy1000_compact.tar.gz"
