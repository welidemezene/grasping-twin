#!/usr/bin/env bash
# Launch stage 23 and walk away. ~45 minutes at 512 envs on the 5070 Ti.
#
#   ./run_stage23.sh            launch
#   ./run_stage23.sh check      is it alive, how far along
#   ./run_stage23.sh judge      run all three verdicts once it is done
#
# DETACHED THE ONLY WAY THAT WORKS HERE. A `docker run` started through an agent
# shell dies with that shell's 10-minute timeout, and `nohup ... & disown` failed
# to create the log at all. setsid with ABSOLUTE paths and stdin from /dev/null
# is the form that survives.

set -u

REPO=/home/$(whoami)/grasping_twin
IMAGE=grasping-twin-isaaclab:latest
LOG=$REPO/week4/stage23_train.log
NAME=stage23

case "${1:-launch}" in

launch)
    # Refuse to start if the source files are older here than on the Windows
    # side -- the WSL copy is where training runs and the Windows copy is where
    # editing happens, and a stale copy would train the wrong config for 45
    # minutes. Docker-written outputs are root-owned so `cp` of *.csv/*.txt
    # partially fails; that is expected, and the source .py files copy fine.
    WIN=/mnt/c/Users/default.LAPTOP-OBNFH8RI/grasping_twin/week4
    if [ -d "$WIN" ]; then
        echo "syncing configs from the Windows copy..."
        cp "$WIN"/stage23_cfg.py "$WIN"/train_stage23.py "$WIN"/probe_envelope.py \
           "$REPO"/week4/ 2>/dev/null
        md5sum "$WIN"/train_stage23.py "$REPO"/week4/train_stage23.py
    fi

    if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
        echo "REFUSING: a container named $NAME is already running." >&2
        exit 1
    fi

    echo "launching stage 23 -> $LOG"
    setsid docker run --rm --gpus all --name "$NAME" \
        -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" \
        train_stage23.py \
        > "$LOG" 2>&1 < /dev/null &

    sleep 5
    echo "launched. check with:  ./run_stage23.sh check"
    ;;

check)
    if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
        echo "RUNNING"
    else
        echo "NOT RUNNING (finished, or never started -- check the log)"
    fi
    echo
    echo "--- registered predictions (written at load, before training) ---"
    sed -n '/PREDICTIONS/,/^$/p' "$REPO"/week4/stage23_reset.txt 2>/dev/null
    echo
    echo "--- newest checkpoints ---"
    ls -t "$REPO"/week4/checkpoints/stage23_* 2>/dev/null | head -3
    echo
    echo "--- log tail (Vulkan errors are always present and are NOT a signal) ---"
    grep -vE 'Warning|deprecated' "$LOG" 2>/dev/null | tail -12
    ;;

judge)
    cd "$REPO"/week4 || exit 1
    run() { docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" "$@"; }

    echo "=== 1. ENVELOPE (primary verdict: did the band widen?) ==="
    run probe_envelope.py checkpoints/stage23_final s23_wide --range 0.15 2>&1 \
        | grep -vE 'Warning|deprecated' | tail -40
    echo
    echo "=== compare against stage 22's map ==="
    sed -n '/SUCCESS/,/SAMPLES/p' envelope_s22_wide.txt
    echo
    echo "=== 2. THE SIX FIXED OFFSETS (>90% and >10 cm) ==="
    ./sweep_shift.sh checkpoints/stage23_final s23
    echo
    echo "=== 3. LIFT SHAPE ACROSS CHECKPOINTS (predictions 2 and 3) ==="
    ./lift_trajectory.sh s23_traj stage23
    ;;

*)
    echo "usage: $0 [launch|check|judge]" >&2
    exit 1
    ;;
esac
