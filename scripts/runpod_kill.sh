#!/usr/bin/env bash
# Emergency stop. `scripts/runpod_kill.sh` lists pods; with an id, terminates it.
set -euo pipefail
KEY=$(cat "${KEY_FILE:-$HOME/.config/runpod/api_key}")
API="https://api.runpod.io/graphql?api_key=$KEY"
if [ $# -eq 0 ]; then
  curl -s -X POST "$API" -H "Content-Type: application/json" \
    -d '{"query":"query { myself { pods { id name desiredStatus costPerHr } } }"}' \
    | jq -r '.data.myself.pods[]? | "\(.id)\t\(.name)\t\(.desiredStatus)\t$\(.costPerHr)/h"'
  echo "(pass a pod id to terminate it)"
else
  curl -s -X POST "$API" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$1\\\"}) }\"}" \
    | jq -r '.errors // "terminated \($1)"' --arg 1 "$1" 2>/dev/null || echo "terminated $1"
fi
