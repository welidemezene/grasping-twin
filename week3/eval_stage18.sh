#!/usr/bin/env bash
# Wait for stage 18 training to finish, then evaluate it properly.
#
# "Properly" per the standing rule: never from one replay, never from cube
# height. check_airborne.py judges 512 parallel trials using the height of the
# cube's LOWEST CORNER, so a cube tipped on a corner can never masquerade as a
# lift. The recorded replay is for the eye, not for the verdict.
#
# Runs each step as its own container: Isaac Lab's simulation context is a
# singleton and will hang if a second scene is built in one process. Uses
# `docker run --rm` without -d, which is the form that actually works here --
# -d aborted at Vulkan init with exit 255, three times running.

set -u

REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest
W3=$REPO/week3
CKPT=week3/checkpoints/stage18_final

run_in_container() {
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week3 "$IMAGE" "$@"
}

echo "=== waiting for stage18 training to finish ==="
while docker ps --format '{{.Names}}' | grep -qx stage18; do
    sleep 60
done
echo "training container is gone at $(date -u +%H:%M:%SZ)"

if [ ! -f "$W3/week3/checkpoints/stage18_final.zip" ]; then
    echo "!! stage18_final.zip MISSING -- training did not reach the end."
    echo "last 20 lines of the training log:"
    tail -20 "$W3/stage18_train.log"
    echo "checkpoints that DO exist:"
    ls -1 "$W3/week3/checkpoints/" | grep '^stage18' | tail -5
    exit 1
fi

echo
echo "=== 512-trial airborne check ==="
run_in_container check_airborne.py "$CKPT" airborne_s18 5 2>&1 \
    | grep -vE 'Warning|\[Error\]|deprecated' | tail -45

echo
echo "=== recorded replay (for the video) ==="
run_in_container record_motion_sticky.py "$CKPT" motion_s18_final.json 5 2>&1 \
    | grep -vE 'Warning|\[Error\]|deprecated' | tail -20

echo
echo "=== DONE ==="
ls -l "$W3"/airborne_s18* "$W3"/motion_s18_final* 2>/dev/null
