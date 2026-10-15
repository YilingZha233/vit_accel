#!/bin/sh
# Read-only environment capture. Redirect to a new run's environment.txt.
set -eu
date -u
uname -a
for f in /proc/cpuinfo /proc/meminfo /etc/os-release /etc/issue; do
    if [ -r "$f" ]; then echo "FILE: $f"; cat "$f"; fi
done
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_* /sys/class/thermal/thermal_zone*/temp; do
    if [ -r "$f" ]; then echo "FILE: $f"; cat "$f"; fi
done
command -v gcc >/dev/null 2>&1 && gcc --version || true
command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 || true
for f in "$@"; do
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$f";
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$f";
    else echo "SHA-256 unavailable; hash copied artifacts on desktop"; fi
done
