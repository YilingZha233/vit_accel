#!/bin/sh
set -eu
cd "$(dirname "$0")"
mode=${1:-pilot}
case "$mode" in
  pilot) reps=3; warmups=1 ;;
  report) reps=20; warmups=3 ;;
  *) echo 'Usage: sh run_board.sh pilot|report'; exit 1 ;;
esac
out="results/$mode"
if [ -e "$out" ]; then
  echo "$out already exists. Preserve it and rename it before repeating."
  exit 1
fi
mkdir -p "$out"
{
  uname -a
  cat /proc/cpuinfo
  cat /proc/meminfo
  cat /etc/issue
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_*; do
    [ -r "$f" ] || continue
    echo "$f"
    cat "$f"
  done
} > "$out/environment.txt" 2>&1
for dir in data/generated/* data/layers/*; do
  [ -d "$dir" ] || continue
  name=${dir##*/}
  case "$dir" in data/generated/*) inner=1000 ;; *) inner=1 ;; esac
  echo "Running $name ($reps samples per contract)..."
  if ! BENCH_SAMPLES_PATH="$out/$name.samples.csv" ./build/bench_matmul \
    "$dir" "$reps" "$inner" "$warmups" > "$out/$name.csv" 2> "$out/$name.log"; then
    cat "$out/$name.log"
    exit 1
  fi
  if [ ! -f "$out/summary.csv" ]; then
    cat "$out/$name.csv" > "$out/summary.csv"
  else
    sed '1d' "$out/$name.csv" >> "$out/summary.csv"
  fi
done
echo "Completed: $out/summary.csv"
