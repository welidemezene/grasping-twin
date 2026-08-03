"""Stage 8 — keep the arm, restart the hand.

The arm is finished: the warm-start policy parks the gripper 3 mm from the cube
centre for most of an episode. The gripper is the opposite. Its output drifted to
a mean of +1.53 (positive means open), so across a 10M run it sampled a close on
3 frames out of 250 — 0.4% of the time. A reward cannot teach an action the
policy has effectively stopped proposing.

So this resets only the gripper row of the action head: weights to zero, bias to
zero. That output then starts at 0 plus exploration noise, meaning close is
proposed about half the time, and the reward decides from there. Every other
action dimension, the shared trunk and the value function are untouched.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage7_4992000_steps")
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

model = PPO.load(args.start_from, env=env, tensorboard_log="week3/logs/stage8")

report = []
with torch.no_grad():
    head = model.policy.action_net
    report.append("gripper bias before %.4f" % float(head.bias[-1]))
    head.weight[-1].zero_()
    head.bias[-1].zero_()
    report.append("gripper bias after  %.4f" % float(head.bias[-1]))
    report.append("gripper weight row max abs after %.6f" % float(head.weight[-1].abs().max()))
    report.append("gripper log_std %.3f -> half of samples should now command close"
                  % float(model.policy.log_std[-1]))
open("stage8_reset.txt", "w").write("\n".join(report) + "\n")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage8",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage8_final")

env.close()
simulation_app.close()
