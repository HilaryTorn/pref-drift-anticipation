#!/usr/bin/env bash

# Launch the independent 4B and 9B DPO pipelines as durable tmux jobs.

set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STORAGE_ROOT=${STORAGE_ROOT:-/ndata/xianglin/ai_drift}
GPU_4B=${GPU_4B:-1}
GPU_9B=${GPU_9B:-2}
LOG_DIR="$STORAGE_ROOT/logs"
PID_DIR="$STORAGE_ROOT/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

launch_one() {
    size=$1
    gpu=$2
    name="dpo-${size,,}"
    pid_file="$PID_DIR/$name.pid"
    log_file="$LOG_DIR/$name.log"
    session="ai-drift-$name"
    if tmux has-session -t "$session" 2>/dev/null; then
        pane_dead=$(tmux display-message -p -t "$session:0.0" '#{pane_dead}')
        if [ "$pane_dead" = 0 ]; then
            pane_pid=$(tmux display-message -p -t "$session:0.0" '#{pane_pid}')
            echo "[launch] $name already running session=$session pid=$pane_pid log=$log_file"
            return
        fi
        exit_status=$(tmux display-message -p -t "$session:0.0" '#{pane_dead_status}')
        echo "[launch] removing exited session=$session status=$exit_status"
        tmux kill-session -t "$session"
    fi
    : >"$log_file"
    printf -v command '%q ' env STORAGE_ROOT="$STORAGE_ROOT" \
        GPU_MEMORY_UTILIZATION=0.50 \
        "$REPO_ROOT/scripts/run_dpo_model.sh" "$size" "$gpu"
    printf -v quoted_log '%q' "$log_file"
    command+=" >>$quoted_log 2>&1"
    tmux new-session -d -s "$session" "$command"
    tmux set-option -t "$session" remain-on-exit on >/dev/null
    pid=$(tmux display-message -p -t "$session:0.0" '#{pane_pid}')
    printf '%s\n' "$pid" >"$pid_file"
    echo "[launch] $name session=$session pid=$pid gpu=$gpu log=$log_file"
}

launch_one 4B "$GPU_4B"
launch_one 9B "$GPU_9B"
