#!/bin/sh
# Run on HPS Linux AFTER compiling model_runner and preparing model/input files.
# Usage: sh scripts/run_hps_model_metrics.sh PARAM BIN INPUTS.tsv NEW_OUT [THREADS]
set -eu
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo 'Usage: sh scripts/run_hps_model_metrics.sh PARAM BIN INPUTS.tsv NEW_OUT [THREADS]' >&2
    exit 2
fi
param=$1; weights=$2; inputs=$3; out=$4; threads=${5:-1}
runner=${MODEL_RUNNER:-./build/model_runner}
precision=${PRECISION:-fp32}
input_blob=${INPUT_BLOB:-in0}; output_blob=${OUTPUT_BLOB:-out0}
runs=${RUNS:-100}; warmups=${WARMUPS:-10}; seconds=${SECONDS_MIN:-0}
if [ -e "$out" ]; then echo "Preserve existing output: $out" >&2; exit 1; fi
mkdir -p "$out"
sh scripts/collect_hps_environment.sh "$runner" "$param" "$weights" "$inputs" > "$out/environment.txt" 2>&1
"$runner" eval "$param" "$weights" "$inputs" "$out/evaluation" "$threads" 0 1 \
    "$input_blob" "$output_blob" 0 "$precision" > "$out/evaluation.stdout" 2> "$out/evaluation.stderr"
for trial in 1 2 3; do
    "$runner" bench "$param" "$weights" "$inputs" "$out/timing_$trial" "$threads" "$warmups" "$runs" \
        "$input_blob" "$output_blob" "$seconds" "$precision" \
        > "$out/timing_$trial.stdout" 2> "$out/timing_$trial.stderr"
done
echo "Saved $out. Validate evaluation logits against desktop before using timing results."
