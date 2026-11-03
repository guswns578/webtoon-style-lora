#!/usr/bin/env bash
set -euo pipefail

B8="$HOME/webtoon_lora/char_run2_clean"
B4="$HOME/webtoon_lora/char_run2_clean_b4"
LOG="$B4/chain_after_b8.log"
mkdir -p "$B4"

{
  echo "[$(date '+%F %T')] chain armed"

  if [[ ! -f "$B8/train.pid" ]]; then
    echo "[$(date '+%F %T')] missing b8 pid file: $B8/train.pid"
    exit 1
  fi

  B8_PID="$(cat "$B8/train.pid")"
  echo "[$(date '+%F %T')] waiting for b8 pid=$B8_PID"

  while kill -0 "$B8_PID" 2>/dev/null; do
    sleep 30
  done

  echo "[$(date '+%F %T')] b8 pid finished"

  if [[ -f "$B4/train.pid" ]] && kill -0 "$(cat "$B4/train.pid")" 2>/dev/null; then
    echo "[$(date '+%F %T')] b4 already running pid=$(cat "$B4/train.pid")"
    exit 0
  fi

  if compgen -G "$B4/output/*.safetensors" > /dev/null; then
    echo "[$(date '+%F %T')] b4 output already exists; not auto-starting"
    ls -lh "$B4/output"/*.safetensors
    exit 0
  fi

  echo "[$(date '+%F %T')] gpu before b4 launch"
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

  echo "[$(date '+%F %T')] launching b4"
  bash "$B4/launch_character_locon_b4.sh"
  echo "[$(date '+%F %T')] b4 launch command completed"
} >> "$LOG" 2>&1
