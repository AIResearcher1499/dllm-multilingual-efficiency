#!/usr/bin/env bash
# Provision a RunPod GPU, run the staged stage-2 plan, fetch results, and
# ALWAYS terminate the pod.
#
# The pod is terminated by an EXIT trap installed the moment it exists, so a
# crash, a Ctrl-C or a failed step still stops the meter. A forgotten A100 is
# $28/day, which is several times the entire experiment budget.
#
# Usage:
#   scripts/runpod_run.sh                 # preflight + K2, then stop and ask
#   scripts/runpod_run.sh --full          # ... and continue into the LEAN grid
#   scripts/runpod_run.sh --keep          # leave the pod up (debugging only)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-NVIDIA A100 80GB PCIe}"  # L40S would be better value but had no supply anywhere
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-20}"          # pod volume; the cache lives on the network volume
CONTAINER_GB="${CONTAINER_GB:-60}"  # local disk: venv goes here, not on a network mount
NETWORK_VOLUME_ID="${NETWORK_VOLUME_ID:-goy4t7g90z}"  # dllm-fert-cache, 60GB, CA-MTL-3 (the only DC with supply)
MAX_HOURS="${MAX_HOURS:-8}"
KEY_FILE="${KEY_FILE:-$HOME/.config/runpod/api_key}"
API="https://api.runpod.io/graphql"

RUN_FULL=0; KEEP=0
for a in "$@"; do
  case "$a" in
    --full) RUN_FULL=1 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown flag $a" >&2; exit 2 ;;
  esac
done

[ -f "$KEY_FILE" ] || { echo "missing $KEY_FILE" >&2; exit 1; }
KEY=$(cat "$KEY_FILE")
POD_ID=""

# Payloads go through a file, not nested command substitution. The nested form
# ($(gql "$(jq ...)")) silently mangled the mutation once already, and a broken
# create is the one failure that can still leave a pod running.
PAYLOAD=$(mktemp)
gql() {  # gql <query-json>
  printf '%s' "$1" > "$PAYLOAD"
  curl -s -X POST "$API?api_key=$KEY" -H "Content-Type: application/json" -d @"$PAYLOAD"
}
gql_file() {  # gql_file <path-to-json>
  curl -s -X POST "$API?api_key=$KEY" -H "Content-Type: application/json" -d @"$1"
}

terminate() {
  local code=$?
  # Results first. A three-hour grid died to a dropped SSH connection and the
  # trap then destroyed the pod holding 500 finished rows. Whatever went wrong,
  # the data comes home before the machine goes away.
  if [ -n "$POD_ID" ] && declare -F fetch_results >/dev/null 2>&1; then
    echo "== rescuing results before terminating =="
    fetch_results || true
    ls -la data/remote/ 2>/dev/null | tail -5
  fi
  # Is the remote job still working? A launcher can die for reasons that have
  # nothing to do with the run -- a dropped ssh under set -e killed this script
  # twice and the trap then destroyed a pod mid-experiment, once at 15 of 22
  # cells. When in doubt, keep the pod: a few idle dollars are recoverable and
  # hours of GPU work are not.
  if [ -n "${POD_ID:-}" ] && [ "$KEEP" -eq 0 ]; then
    REMOTE_STATE=$($SSH 'pgrep -f "phase5.sh|phase4_p3.sh|phase4_canvas.sh|phase4_gateA.sh|phase3_sweep.sh|dllmfert g0" >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNKNOWN)
    if [ "$REMOTE_STATE" != "STOP" ]; then
      echo ""
      echo "== POD LEFT RUNNING ($REMOTE_STATE) =="
      echo "   The launcher is exiting but the remote job is not finished."
      echo "   Pod $POD_ID is still up and still billing. Terminate it yourself"
      echo "   once the work is done:"
      echo "     bash scripts/runpod_kill.sh $POD_ID"
      KEEP=1
    fi
  fi
  if [ -n "$POD_ID" ] && [ "$KEEP" -eq 0 ]; then
    echo ""
    echo "== terminating pod $POD_ID =="
    gql "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$POD_ID\\\"}) }\"}" \
      | jq -r '.errors // "terminated"'
  elif [ -n "$POD_ID" ]; then
    echo "!! pod $POD_ID LEFT RUNNING (--keep). Stop it with:"
    echo "   scripts/runpod_kill.sh $POD_ID"
  fi
  exit $code
}

# Supply is the binding constraint, not price. A network volume pins the pod to
# one data centre, and CA-MTL-3 went from "A100 available" to every card
# returning SUPPLY_CONSTRAINT inside an hour. So: try a preference list, and if
# the volume's data centre is dry, fall back to running anywhere and paying the
# re-download. A failed create costs nothing, which makes this cheap to attempt.
GPU_PREFS=(${GPU_PREFS:-"NVIDIA L40S" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA RTX A6000" "NVIDIA A40" "NVIDIA H100 80GB HBM3"})
ATTEMPTS="${ATTEMPTS:-10}"
RETRY_SLEEP="${RETRY_SLEEP:-60}"

try_create() {  # try_create <gpu> <volume-id-or-empty>
  local gpu="$1" vol="$2" json
  json=$(jq -n --arg gpu "$gpu" --arg img "$IMAGE" --arg disk "$DISK_GB" \
    --arg cdisk "$CONTAINER_GB" --arg vol "$vol" \
    '{query: ("mutation { podFindAndDeployOnDemand(input: {cloudType: SECURE, gpuCount: 1"
      + ", volumeInGb: " + $disk
      + ", containerDiskInGb: " + $cdisk
      + ", minVcpuCount: 8, minMemoryInGb: 32"
      + (if $vol == "" then "" else ", networkVolumeId: " + ($vol|tojson) end)
      + ", gpuTypeId: " + ($gpu|tojson)
      + ", name: \"dllm-fertility\""
      + ", imageName: " + ($img|tojson)
      + ", supportPublicIp: true, startSsh: true"
      + ", dockerArgs: \"\", ports: \"22/tcp\""
      + ", volumeMountPath: " + (if $vol == "" then "\"/workspace\"" else "\"/runpod-volume\"" end)
      + "}) { id } }")}')
  echo "$json" > "$CREATE_JSON"
  gql_file "$CREATE_JSON" | jq -r '.data.podFindAndDeployOnDemand.id // empty'
}

CREATE_JSON=$(mktemp)
trap terminate EXIT INT TERM
POD_ID=""; USED_VOLUME=""
for attempt in $(seq 1 "$ATTEMPTS"); do
  for vol in "$NETWORK_VOLUME_ID" ""; do
    for gpu in "${GPU_PREFS[@]}"; do
      POD_ID=$(try_create "$gpu" "$vol")
      if [ -n "$POD_ID" ]; then
        GPU="$gpu"; USED_VOLUME="$vol"
        echo "== pod $POD_ID: $gpu, $([ -n "$vol" ] && echo "cache volume $vol" || echo 'NO volume - weights will be re-downloaded') =="
        break 3
      fi
    done
    [ -n "$NETWORK_VOLUME_ID" ] && [ -z "$vol" ] || continue
  done
  echo "attempt $attempt/$ATTEMPTS: nothing available anywhere; sleeping ${RETRY_SLEEP}s"
  sleep "$RETRY_SLEEP"
done
if [ -z "$POD_ID" ]; then
  echo "no capacity after $ATTEMPTS attempts across ${#GPU_PREFS[@]} GPU types" >&2
  exit 1
fi

echo "== waiting for the pod to report an SSH port =="
HOST=""; PORT=""
for _ in $(seq 1 60); do
  R=$(gql "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip publicPort privatePort isIpPublic } } } }\"}")
  HOST=$(echo "$R" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic) | .ip' | head -1)
  PORT=$(echo "$R" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic) | .publicPort' | head -1)
  [ -n "$HOST" ] && [ -n "$PORT" ] && break
  sleep 10
done
[ -n "$HOST" ] || { echo "SSH never came up" >&2; exit 1; }
echo "ssh root@$HOST -p $PORT"

HF_DIR=$([ -n "$USED_VOLUME" ] && echo /runpod-volume/hf || echo /root/hf)
mkdir -p data/remote
fetch_results() {
  # Plain cat per file, not a tar pipe. The tar version failed silently on
  # macOS and `2>/dev/null || true` hid it, so 544 finished rows sat on a pod
  # while the local copy stayed empty. Small files; simplicity is worth more
  # than the round trips.
  local f ok=0
  for f in g0.jsonl g0_llada.jsonl phase3.jsonl phase3_summary.json \
           llada_preflight.jsonl ablation.jsonl \
           preflight.jsonl k2.jsonl canvas.json fertility.json \
           g0_summary.json k2_summary.json; do
    if $SSH "test -s /root/dllm-fertility/data/$f" 2>/dev/null; then
      $SSH "cat /root/dllm-fertility/data/$f" > "data/remote/$f" 2>/dev/null \
        && ok=$((ok + 1))
    fi
  done
  echo "  fetched $ok file(s); g0.jsonl now $(wc -l < data/remote/g0.jsonl 2>/dev/null || echo 0) rows"
}
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes -p $PORT root@$HOST"

# The API reports the SSH port as soon as the pod is RUNNING, but the container
# is still pulling a ~20 GB image at that point and sshd is not listening yet.
# The previous version polled for 150s, fell through silently when that ran out,
# and handed a dead host to rsync. Wait properly, and fail loudly if it never
# comes up so the trap can stop the meter.
echo "== waiting for sshd (up to ${SSH_WAIT_MIN:-8} min; the image pull dominates) =="
SSH_READY=0
for i in $(seq 1 $(( ${SSH_WAIT_MIN:-8} * 4 ))); do
  if $SSH true 2>/dev/null; then SSH_READY=1; echo "sshd up after $(( i * 15 ))s"; break; fi
  [ $(( i % 4 )) -eq 0 ] && echo "  ... $(( i * 15 ))s"
  sleep 15
done
if [ "$SSH_READY" -ne 1 ]; then
  echo "sshd never accepted a connection on $HOST:$PORT" >&2
  echo "pod state:" >&2
  gql "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { desiredStatus runtime { uptimeInSeconds } } }\"}" >&2
  exit 1
fi

# tar over ssh, not rsync: the runpod/pytorch image has no rsync, and the repo
# is small enough that a stream costs nothing. This also removes a dependency
# on whatever happens to be installed in the image.
echo "== sending repo (tar over ssh) =="
# COPYFILE_DISABLE stops macOS writing AppleDouble ._ files into the stream,
# and --no-same-owner stops the remote tar trying to chown to a uid that does
# not exist there. Both only ever produced noise, but under `set -e` the
# non-zero exit killed a transfer that had actually succeeded.
COPYFILE_DISABLE=1 tar czf - \
    --exclude='._*' --exclude='.DS_Store' \
    --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='.git' --exclude='data/*.jsonl' --exclude='data/remote' . \
  | $SSH 'mkdir -p /root/dllm-fertility && tar xzf - --no-same-owner -C /root/dllm-fertility && echo "--- sent ---" && ls /root/dllm-fertility'


# HF_HOME on the network volume: 46 GB of weights, large sequential files, and
# it must survive the pod -- re-downloading them ten times cost more than every
# experiment run put together.
# The venv on local container disk: thousands of small files, which is exactly
# what a network mount is worst at (20 minutes to build one on /workspace).
REMOTE_BASE="export PATH=\$HOME/.local/bin:\$PATH PYTHONUNBUFFERED=1 \
  HF_HOME=$HF_DIR UV_PROJECT_ENVIRONMENT=/root/venv \
  && mkdir -p $HF_DIR && cd /root/dllm-fertility"
echo "== bootstrap =="
$SSH "$REMOTE_BASE && bash scripts/bootstrap_gpu.sh" 2>&1 | tail -30

if [ "${P5:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 5: throughput vs quality, 11 languages x 2 models =="
  # P5_ARMS/P5_LANGS/P5_N are set on the laptop and must be carried across the
  # ssh boundary explicitly -- a local export does not reach the remote shell,
  # and the run silently reverts to defaults. That cost 80 minutes redoing an
  # arm that was already finished.
  $SSH "$REMOTE_BASE && P5_ARMS='${P5_ARMS:-llada dream}' P5_LANGS='${P5_LANGS:-en zh es fr de ja sw ru th bn te}' P5_N='${P5_N:-100}' nohup timeout ${MAX_HOURS}h bash scripts/phase5.sh > phase5.log 2>&1 & echo started"
  sleep 20
  # An unreachable box is not evidence the job finished. One failed check is
  # noise; only a run of consecutive STOPs means done. Treating a single ssh
  # hiccup as completion terminated a pod mid-run at 15 of 22 cells.
  misses=0
  while :; do
    st=$($SSH 'pgrep -f phase5.sh >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN)  misses=0 ;;
      STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *)    echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 180
    $SSH "grep -E '^=====|per second' /root/dllm-fertility/phase5.log | tail -2" 2>/dev/null | tr -d '\r'
    mkdir -p data/remote/phase5
    $SSH "cd /root/dllm-fertility/data && tar czf - phase5" > data/remote/phase5/phase5.tgz 2>/dev/null
    echo "  --- fetched $(wc -c < data/remote/phase5/phase5.tgz 2>/dev/null || echo 0) bytes"
  done
  mkdir -p data/remote/phase5
  $SSH "cat /root/dllm-fertility/phase5.log" > data/remote/phase5/phase5.log 2>/dev/null
  $SSH "cd /root/dllm-fertility/data && tar czf - phase5" > data/remote/phase5/phase5.tgz 2>/dev/null
  $SSH "tail -30 /root/dllm-fertility/phase5.log" 2>&1 | tr -d '\r'
  exit 0
fi

if [ "${P3:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 4 P3: quality-matched speed-up across languages =="
  $SSH "$REMOTE_BASE && nohup timeout ${MAX_HOURS}h bash scripts/phase4_p3.sh > p3.log 2>&1 & echo started"
  sleep 20
  misses=0
  while :; do
    st=$($SSH 'pgrep -f phase4_p3.sh >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN) misses=0 ;; STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *) echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 180
    $SSH "grep -E '^===== cell|^Total NFE|^\\|mgsm' /root/dllm-fertility/p3.log | tail -2" 2>/dev/null | tr -d '\r'
    mkdir -p data/remote/p3
    $SSH "cd /root/dllm-fertility/data && tar czf - p3" > data/remote/p3/p3.tgz 2>/dev/null
    echo "  --- fetched $(wc -c < data/remote/p3/p3.tgz 2>/dev/null || echo 0) bytes"
  done
  mkdir -p data/remote/p3
  $SSH "cat /root/dllm-fertility/p3.log" > data/remote/p3/p3.log 2>/dev/null
  $SSH "cd /root/dllm-fertility/data && tar czf - p3" > data/remote/p3/p3.tgz 2>/dev/null
  $SSH "tail -40 /root/dllm-fertility/p3.log" 2>&1 | tr -d '\r'
  exit 0
fi

if [ "${CANVAS:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 4b: canvas sweep =="
  $SSH "$REMOTE_BASE && nohup timeout ${MAX_HOURS}h bash scripts/phase4_canvas.sh > canvas.log 2>&1 & echo started"
  sleep 20
  misses=0
  while :; do
    st=$($SSH 'pgrep -f phase4_canvas.sh >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN) misses=0 ;; STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *) echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 120
    $SSH "tail -2 /root/dllm-fertility/canvas.log" 2>/dev/null | tr -d '\r'
    echo "  ---"
  done
  mkdir -p data/remote/canvas
  $SSH "cat /root/dllm-fertility/canvas.log" > data/remote/canvas/canvas.log 2>/dev/null
  # tar the per-item samples back; they are the part that cannot be recomputed
  $SSH "cd /root/dllm-fertility/data && tar czf - canvas_runs" > data/remote/canvas/canvas_runs.tgz 2>/dev/null
  echo "fetched: $(wc -c < data/remote/canvas/canvas_runs.tgz 2>/dev/null || echo 0) bytes of samples"
  $SSH "tail -40 /root/dllm-fertility/canvas.log" 2>&1 | tr -d '\r'
  exit 0
fi

if [ "${GATEA:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 4 Gate A: reproduce Fast-dLLM's published accuracy =="
  $SSH "$REMOTE_BASE && nohup timeout ${MAX_HOURS}h bash scripts/phase4_gateA.sh > gateA.log 2>&1 & echo started"
  sleep 20
  misses=0
  while :; do
    st=$($SSH 'pgrep -f phase4_gateA.sh >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN) misses=0 ;; STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *) echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 120
    $SSH "tail -3 /root/dllm-fertility/gateA.log" 2>/dev/null | tr -d '\r'
    echo "  ---"
  done
  $SSH "cat /root/dllm-fertility/gateA.log" > data/remote/gateA.log 2>/dev/null
  $SSH "tail -80 /root/dllm-fertility/gateA.log" 2>&1 | tr -d '\r'
  echo "full log in data/remote/gateA.log"
  exit 0
fi

if [ "${RECON:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 4 stage 0: Fast-dLLM recon =="
  # Cheap and read-only. Nothing here is a measurement; it exists so the
  # measuring stages are not written against a guessed API.
  $SSH "$REMOTE_BASE && bash scripts/phase4_recon.sh" 2>&1 | tr -d '\r'
  echo ""
  echo "recon done -- pod left RUNNING on purpose; terminate it yourself"
  exit 0
fi

if [ "${SWEEP:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 3: repetition-penalty sweep =="
  # Poll on the sweep script, not on `dllmfert g0`: between penalty settings
  # there is no g0 process, and a loop watching for one would decide the run
  # had finished and terminate the pod mid-sweep.
  $SSH "$REMOTE_BASE && nohup timeout ${MAX_HOURS}h bash scripts/phase3_sweep.sh > sweep.log 2>&1 & echo started"
  sleep 20
  misses=0
  while :; do
    st=$($SSH 'pgrep -f phase3_sweep.sh >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN) misses=0 ;; STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *) echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 120
    fetch_results
    $SSH "grep -E '^(===|  rows now)' /root/dllm-fertility/sweep.log | tail -2" 2>/dev/null | tr -d '\r'
  done
  $SSH "tail -60 /root/dllm-fertility/sweep.log" 2>&1 | tr -d '\r' | tail -60
  fetch_results
  echo "results in data/remote/"
  exit 0
fi

if [ "${GRID:-0}" = "1" ]; then
  echo ""
  echo "== PHASE 2: grid =="
  # G0_ARGS lets a run target a different arm, fertility file or canvas table
  # without editing this script. Default is the Dream operating point.
  G0_ARGS="${G0_ARGS:---n ${GRID_N:-50} --modes threshold --canvas-mode measured --dllm-models Dream-org/Dream-v0-Instruct-7B --shots 3 --keep-text --resume --out data/g0.jsonl}"
  OUT_FILE="${OUT_FILE:-g0.jsonl}"
  $SSH "$REMOTE_BASE && nohup timeout ${MAX_HOURS}h uv run dllmfert g0 $G0_ARGS > grid.log 2>&1 & echo started"
  misses=0
  while :; do
    st=$($SSH 'pgrep -f "dllmfert g0" >/dev/null && echo RUN || echo STOP' 2>/dev/null || echo UNREACHABLE)
    case "$st" in
      RUN) misses=0 ;; STOP) misses=$((misses+1)); [ "$misses" -ge 3 ] && break ;;
      *) echo "  (ssh unreachable -- not counted as finished)" ;;
    esac
    sleep 120
    fetch_results     # pull rows as they land, so a crash costs minutes not hours
    $SSH "tail -2 /root/dllm-fertility/grid.log" 2>/dev/null | tr -d '\r' | tail -1
  done
  $SSH "tail -20 /root/dllm-fertility/grid.log" 2>&1 | tail -20
  fetch_results
  $SSH "$REMOTE_BASE && uv run dllmfert verdict --rows data/$OUT_FILE" 2>&1 | tail -40 || true
  fetch_results
  echo "results in data/remote/"
  exit 0
fi

echo ""
echo "== 1/2 preflight (validity, not just plumbing) =="
# --keep-text is the point: last time "0 errors" hid three models being fed a
# prompt they were never trained to read. Three languages spanning the
# fertility range, threshold mode only -- naive is analytically 1 token/step
# and at Telugu's canvas would cost more than the rest of the run combined.
# Dream and the AR baseline only. LLaDA is the optional second diffusion arm
# and is not in the LEAN grid; pulling its 16 GB into preflight wasted half an
# hour and a third of a pod for an arm nobody was about to run.
$SSH "$REMOTE_BASE && timeout ${MAX_HOURS}h uv run dllmfert g0 \
    --langs en th te --n 4 --modes threshold --base-canvas 512 \
    --dllm-models Dream-org/Dream-v0-Instruct-7B \
    --shots 3 --keep-text --out data/preflight.jsonl" 2>&1 | tail -30

echo ""
echo "== 2/2 K2 (mechanism — this can kill the design) =="
$SSH "$REMOTE_BASE && timeout ${MAX_HOURS}h uv run dllmfert k2 --lang en --canvases 256 512 1024 2048" 2>&1 | tail -30

mkdir -p data/remote
fetch_results
K2_DECISION=$(jq -r '.decision // "UNKNOWN"' data/remote/k2_summary.json 2>/dev/null || echo UNKNOWN)
echo ""
echo "===================== K2 decision: $K2_DECISION ====================="

if [ "$K2_DECISION" != "PROCEED" ]; then
  echo "Not proceeding to the grid. See data/remote/k2_summary.json."
  exit 0
fi
if [ "$RUN_FULL" -ne 1 ]; then
  echo "K2 cleared. Re-run with --full to continue into the LEAN grid."
  exit 0
fi

echo ""
echo "== LEAN grid: Dream-7B threshold vs Qwen2.5-7B, 11 languages =="
$SSH "$REMOTE_BASE && timeout ${MAX_HOURS}h uv run dllmfert g0 --modes threshold --resume" 2>&1 | tail -30
$SSH "$REMOTE_BASE && uv run dllmfert verdict" 2>&1 | tail -40
fetch_results
echo "results in data/remote/"
