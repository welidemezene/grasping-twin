#!/usr/bin/env bash
# Drive bench_envs.py once per env count, each in its own container.
#
# Isaac Lab's simulation context is a singleton: closing an env and building
# another in the same process hangs forever (observed 21 min at 0% GPU). So the
# loop lives out here, not inside the Python.
#
# Usage:  ./bench_envs.sh 1024 2048 4096

set -u

REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest
OUT=$REPO/week3/bench_envs.txt
STEPS=${STEPS:-300}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-900}

COUNTS=("$@")
if [ ${#COUNTS[@]} -eq 0 ]; then
    COUNTS=(1024 2048 4096)
fi

for N in "${COUNTS[@]}"; do
    echo "########## num_envs=$N ##########"
    timeout "$PER_RUN_TIMEOUT" docker run --rm --gpus all \
        -v "$REPO":/workspace -w /workspace/week3 \
        "$IMAGE" bench_envs.py --envs "$N" --steps "$STEPS" 2>&1 \
        | grep -E '\[bench\]|rror|CUDA|out of memory' \
        | tail -15
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ]; then
        echo "TIMED OUT after ${PER_RUN_TIMEOUT}s -- recording as hang"
        printf '%8d %12s %10s %14s  HANG\n' "$N" "-" "-" "-" >> "$OUT"
    fi
    echo "exit=$rc"
done

echo "===== TABLE ====="
printf '%8s %12s %10s %14s  %s\n' envs steps/s GB 10M-run status
cat "$OUT" 2>/dev/null
