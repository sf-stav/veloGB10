#!/bin/bash
# Persistent host-memory trace: one line per second (MemAvailable / SwapFree / top process RSS)
# into a file that survives a crash. Run in the background during any large-model test:
#   scripts/memlog.sh /var/tmp/memlog.txt &
OUT=${1:-/var/tmp/memlog.txt}
while true; do
  avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  swap=$(awk '/SwapFree/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  top=$(ps -eo rss,comm --sort=-rss | awk 'NR==2 {printf "%s %.1fGB", $2, $1/1048576}')
  echo "$(date +%H:%M:%S) avail=${avail}GB swapfree=${swap}GB top=${top}" >> "$OUT"
  sleep 1
done
