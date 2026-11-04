"""Count how far each failed episode got, from lerobot-eval's eval_info.json.

Stage = the episode's max reward (gym_aloha transfer cube, tasks/sim.py):
  0 reach   (right gripper never touched the cube)
  1 grasp   (touched, cube never left the table)
  2 carry   (lifted, left gripper never touched it)
  3 handover (left gripper touched, cube not held in the air)
  4 success

Usage: python week8/tools/count_failures.py <eval_info.json> [more files...]
(written by Claude, at Woldemedihn's request)
"""
import json
import sys

STAGES = ["reach", "grasp", "carry", "handover"]

for path in sys.argv[1:]:
    metrics = json.load(open(path))["per_task"][0]["metrics"]
    counts = [0, 0, 0, 0]
    for success, max_reward in zip(metrics["successes"], metrics["max_rewards"]):
        if not success:
            counts[int(max_reward)] += 1
    n = len(metrics["successes"])
    summary = "  ".join(f"{s}={c}" for s, c in zip(STAGES, counts))
    print(f"{path}\n  episodes={n}  failures={sum(counts)}  {summary}")
