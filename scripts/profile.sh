#!/usr/bin/env bash
# Profile the engine, using the deepest analysis this machine will allow.
#
# GPU performance counters (what `ncu` reads) are LOCKED BY DEFAULT on every NVIDIA machine --
# they are a side channel, so the driver restricts them to root. See PROFILING.md.
#
# This script therefore DETECTS what is available and degrades honestly:
#
#   counters available  -> timeline + per-shape bandwidth + stall/occupancy breakdown
#   counters locked     -> timeline + per-shape bandwidth + roofline arithmetic, and it SAYS SO
#
# Nothing we ship ever REQUIRES the permission. Counters are a developer convenience for kernel
# work, never a runtime dependency, and every published number must be reproducible without them.
#
# Usage:  scripts/profile.sh <model-dir> [kernel-name]
set -uo pipefail

MODEL=${1:?usage: scripts/profile.sh <model-dir> [kernel-name]}
KERNEL=${2:-gemm_mma_fp4_b}
BIN=${BIN:-./target/release/gb10_inference}
OUT=${OUT:-$(mktemp -d)}

[ -x "$BIN" ] || { echo "no binary at $BIN (cargo build --release)"; exit 1; }

# ---------------------------------------------------------------- capability probe
# The probe MUST profile a kernel the command actually launches, or ncu never attempts a counter
# read, never errors, and we wrongly conclude counters work. `bw_read_b` is the one kernel
# --probe-bandwidth is guaranteed to run, so profile exactly that.
counters=no
if command -v ncu >/dev/null 2>&1; then
  ncu --kernel-name bw_read_b --launch-count 1 --metrics sm__cycles_elapsed.avg \
      "$BIN" --probe-bandwidth --model-dir "$MODEL" >"$OUT/probe.log" 2>&1
  if ! grep -q "ERR_NVGPUCTRPERM" "$OUT/probe.log" && grep -q "sm__cycles_elapsed" "$OUT/probe.log"; then
    counters=yes
  fi
fi

echo "==================================================================="
echo " GPU performance counters: $( [ $counters = yes ] && echo "AVAILABLE" || echo "LOCKED (ERR_NVGPUCTRPERM)" )"
if [ $counters = no ]; then
cat <<'EOF'
   -> Running timeline + roofline analysis only. This is enough for "which kernel
      dominates" and "what bandwidth is each shape achieving", but NOT for "why".
   -> To unlock (optional; see PROFILING.md §2 for why NVIDIA locks it):
        one-off :  sudo /usr/local/cuda/bin/ncu --set full -o /tmp/prof <binary> <args>
                   (absolute path: sudo resets PATH, so bare `sudo ncu` = command not found)
        forever :  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
                     | sudo tee /etc/modprobe.d/nvidia-profiling.conf   # then reboot
EOF
fi
echo "==================================================================="

# ---------------------------------------------------------------- 1. the roofline
echo
echo "--- 1. THE ROOFLINE (measure it; never quote a spec sheet) ---"
"$BIN" --probe-bandwidth --model-dir "$MODEL" 2>/dev/null | grep -E "GB/s|PEAK"

# ---------------------------------------------------------------- 2. timeline
echo
echo "--- 2. WHERE THE TIME GOES ---"
nsys profile -t cuda -o "$OUT/p" --force-overwrite true \
  "$BIN" --bench-mtp --model-dir "$MODEL" --depth 2 --max-new-tokens 40 >/dev/null 2>&1
nsys stats --force-export=true --report cuda_gpu_kern_sum --format column "$OUT/p.nsys-rep" 2>/dev/null \
  | grep -vE "^Processing|^Generating|WARNING|^ *File|^ *Use|^$" | head -10

# ---------------------------------------------------------------- 3. per-shape bandwidth
echo
echo "--- 3. ACHIEVED BANDWIDTH PER GEMM SHAPE (vs the roofline above) ---"
nsys stats --force-export=true --report cuda_gpu_trace --format csv "$OUT/p.nsys-rep" 2>/dev/null \
 | grep "$KERNEL" \
 | awk -F',' '{ g=$4; cnt[g]++; sum[g]+=$2 }
   END { for (k in cnt) printf "  grid=%-7s M=%-7d launches=%-6d avg=%8.1f us  total=%8.2f ms\n",
                                k, k*16, cnt[k], sum[k]/cnt[k]/1000, sum[k]/1e6 }' | sort -t= -k2 -n

# ---------------------------------------------------------------- 4. launch gaps
echo
echo "--- 4. GPU BUSY vs IDLE (decides whether CUDA graphs are worth anything) ---"
# KERNEL rows only. The trace also carries memcpy/memset rows with a different column layout;
# counting those as "kernels" turns the idle figure into garbage (it read 83% once).
nsys stats --force-export=true --report cuda_gpu_trace --format csv "$OUT/p.nsys-rep" 2>/dev/null \
 | grep -vE "memcpy|memset|^Start|^Generating|^Processing" \
 | awk -F',' '$1+0>0 && $2+0>0 { s=$1+0; d=$2+0; if (prev>0 && s>prev) gap += s-prev;
                                  busy += d; if (s+d>prev) prev=s+d }
   END { printf "  busy %.1f ms | gaps %.1f ms (%.1f%% of span)\n", busy/1e6, gap/1e6, 100*gap/(busy+gap) }'

# ---------------------------------------------------------------- 5. the WHY (counters only)
if [ $counters = yes ]; then
  echo
  echo "--- 5. WHY (hardware counters) — $KERNEL ---"
  ncu --kernel-name "$KERNEL" --launch-skip 300 --launch-count 4 \
      --metrics smsp__issue_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
      "$BIN" --bench-mtp --model-dir "$MODEL" --depth 2 --max-new-tokens 4 2>&1 \
    | grep -E "issue_active|warps_active|dram_throughput|sectors|stalled" | sed 's/^/  /'
  echo
  echo "  Reading it:  low issue_active + high long_scoreboard  => DRAM-latency bound (need more"
  echo "               warps in flight).  high barrier => the epilogue reduction is the cost."
  echo "               low warps_active => occupancy/wave-quantization is real."
else
  echo
  echo "--- 5. WHY — SKIPPED (counters locked). See the banner above. ---"
fi

echo
echo "artifacts: $OUT"
