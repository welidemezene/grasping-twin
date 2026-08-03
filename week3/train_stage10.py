"""Stage 10 — the decoupled reward, and exploration that cannot go extinct.

Start from stage 7's arm (the best ever recorded: 224 of 249 frames inside
3 cm, closest 2.9 mm) with the gripper head zeroed again — that part of the
stage-8/9 surgery was right. What was missing:

1. The reward. Under the old shaping, PPO *chose* open with close proposed
   half the time (grip bias climbed +0.017 -> +0.062 over 3M steps). The new
   grasp_reward makes approach, close-at-cube, and lift a monotonic staircase.
2. The entropy. ent_coef was 0.0, so the gripper's exploration collapsed to
   sigma ~0.4 by stage 7 and close went extinct (sampled 0.4% of frames).
   0.01 keeps a floor under exploration for the whole run — the Day 9 lesson.
3. The spread. Stage 7's gripper log_std (-0.93) is restored to -0.5 so close
   is proposed roughly half the time from the first rollout.

The value function still expects the old reward and will thrash for the first
rollouts; that is PPO relearning what states are worth, not a bug.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage7_4992000_steps")
parser.add_argument("--ent_coef", type=float, default=0.01)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)

# custom_objects overrides the value stored in the checkpoint — without it,
# load() would silently restore ent_coef=0.0 and the fix would be a no-op
model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage10",
    custom_objects={"ent_coef": args.ent_coef},
)

report = []
with torch.no_grad():
    head = model.policy.action_net
    report.append("gripper bias before %.4f" % float(head.bias[-1]))
    head.weight[-1].zero_()
    head.bias[-1].zero_()
    report.append("gripper bias after  %.4f" % float(head.bias[-1]))
    report.append("gripper weight row max abs after %.6f" % float(head.weight[-1].abs().max()))
    report.append("gripper log_std before %.3f" % float(model.policy.log_std[-1]))
    model.policy.log_std[-1] = -0.5
    report.append("gripper log_std reset to %.3f -> close proposed ~half the time" % float(model.policy.log_std[-1]))
    report.append("ent_coef %.4f (was 0.0 through stage 9 — exploration can no longer go extinct)" % model.ent_coef)
open("stage10_reset.txt", "w").write("\n".join(report) + "\n")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage10",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage10_final")

env.close()
simulation_app.close()
