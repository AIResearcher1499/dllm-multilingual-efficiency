#!/usr/bin/env bash
# Machine-agnostic cell runner. Replaces the RunPod-specific launcher for a box
# we own: no provisioning, no terminate trap, no network volume.
#
#   CELLS=cells.txt MODE=parallel GPUS=0,1 bash scripts/run_local.sh
#
# CELLS is a file of shell commands, one per line, blank lines and # ignored.
# Each line is one cell and gets its own log under LOGDIR.
#
# MODE=parallel  -- one cell per GPU at a time. Correct for anything scored on
#                   NFE, accuracy or text: those are hardware-invariant and a
#                   neighbour on the box cannot change them.
# MODE=serial    -- one cell at a time on the whole box. **Required** for
#                   anything scored on wall_clock or tokens/second: two cells
#                   contend for CPU and PCIe, and the contention lands directly
#                   in the number being measured.
#
# Rows carry `timing_trustworthy` from dllmfert.provenance, so a parallel run
# that later gets analysed for timing is detectable rather than silently wrong.
# This script sets the flag's ground truth; it does not replace checking it.
set -uo pipefail
cd "$(dirname "$0")/.."

CELLS="${CELLS:?set CELLS=<file of one command per line>}"
MODE="${MODE:-serial}"
GPUS="${GPUS:-0}"
LOGDIR="${LOGDIR:-logs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGDIR"

# Portable read: macOS ships bash 3.2, which has neither mapfile nor
# associative arrays. Keeping this runnable on the laptop is what lets it be
# tested before it touches the GPU box.
LINES=()
while IFS= read -r _l; do
  case "$_l" in ''|\#*) continue;; esac
  LINES+=("$_l")
done < "$CELLS"
echo "cells: ${#LINES[@]}   mode: $MODE   gpus: $GPUS   logs: $LOGDIR"
[ "${#LINES[@]}" -gt 0 ] || { echo "nothing to run" >&2; exit 1; }

IFS=',' read -r -a GPUARR <<< "$GPUS"
if [ "$MODE" = "serial" ]; then
  echo "serial mode: timing-sensitive work, one cell at a time on GPU ${GPUARR[0]}"
fi

run_one () {
  local idx="$1" gpu="$2" cmd="$3"
  local log
  log="$LOGDIR/cell$(printf '%03d' "$idx").log"
  {
    echo "### cell $idx on GPU $gpu"
    echo "### $cmd"
    echo "### started $(date +%Y-%m-%dT%H:%M:%S%z)"
  } > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >> "$log" 2>&1
  local rc=$?
  echo "### exit $rc at $(date +%Y-%m-%dT%H:%M:%S%z)" >> "$log"
  echo "cell $idx (gpu $gpu) exit=$rc  -> $log"
  return $rc
}

fails=0
if [ "$MODE" = "serial" ]; then
  for i in "${!LINES[@]}"; do
    run_one "$i" "${GPUARR[0]}" "${LINES[$i]}" || fails=$((fails+1))
  done
else
  # One slot per GPU, refilled as soon as its cell exits. Parallel indexed
  # arrays rather than an associative one, for bash 3.2.
  SLOT_PID=()
  for _i in "${!GPUARR[@]}"; do SLOT_PID[$_i]=""; done
  next=0; live=1
  while [ "$next" -lt "${#LINES[@]}" ] || [ "$live" -gt 0 ]; do
    for _i in "${!GPUARR[@]}"; do
      if [ -z "${SLOT_PID[$_i]}" ] && [ "$next" -lt "${#LINES[@]}" ]; then
        run_one "$next" "${GPUARR[$_i]}" "${LINES[$next]}" &
        SLOT_PID[$_i]=$!
        next=$((next+1))
      fi
    done
    sleep 2
    live=0
    for _i in "${!GPUARR[@]}"; do
      pid="${SLOT_PID[$_i]}"
      if [ -n "$pid" ]; then
        if kill -0 "$pid" 2>/dev/null; then
          live=$((live+1))
        else
          wait "$pid" || fails=$((fails+1))
          SLOT_PID[$_i]=""
        fi
      fi
    done
  done
fi

echo ""
echo "done: ${#LINES[@]} cells, $fails failed. logs in $LOGDIR"
exit $(( fails > 0 ))
