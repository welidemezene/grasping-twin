#!/bin/bash
# Re-record three checkpoints with gripper/base/done tracking.
R=/mnt/c/Users/default.LAPTOP-OBNFH8RI/grasping_twin
IMG=grasping-twin-isaaclab:latest

run() {
  ckpt=$1
  out=$2
  docker run --rm --gpus all -v "$R:/workspace" -w /workspace/week3 "$IMG" \
    record_motion.py "week3/checkpoints/$ckpt" "$out" > /tmp/rec_$out.log 2>&1
  echo "=== $ckpt -> $out"
  base=${out%.json}
  if [ -f "$R/week3/${base}_summary.txt" ]; then
    cat "$R/week3/${base}_summary.txt"
  else
    echo "NO SUMMARY — tail of log:"
    tail -20 /tmp/rec_$out.log
  fi
}

run stage1_199680_steps motion_199680.json
run stage1_998400_steps motion_998400.json
run stage1_final        motion.json
