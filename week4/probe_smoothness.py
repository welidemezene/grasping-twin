"""How much does the arm shake, and is it worse than it used to be?

The vibration is visible in stage 22's footage. Two candidate causes, and they
have different fixes:

  A. EXPLORATION NOISE -- ruled out before running anything. Replays use
     deterministic=True, so no action is sampled; the policy emits one value per
     observation. Whatever shakes is the policy REACTING, not dithering.

  B. NOTHING PAYS FOR SMOOTHNESS. curriculum_lift_cfg.py kills the stock
     curriculum that escalates action_rate and joint_vel from -1e-4 to -1e-1,
     because that 1000x tax at ~5.1M steps collapsed stages 10 and 11 and froze
     every long run. The terms were left at their base weight of -1e-4 -- four
     orders of magnitude below the task rewards, which run 2 to 12. So there is
     a smoothness penalty in name only.

This measures it instead of arguing about it. Per step it records the change in
commanded joint targets (the action delta) and the arm's joint speeds, then
reports medians and the 95th percentile. Run it on two checkpoints to compare:
stage 16 trained at a fixed spawn, stage 22 at a randomized one.

    probe_smoothness.py <checkpoint> <label>

Medians, not means: one flung frame moves a mean and tells you nothing about
what the motion looks like for most of an episode.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("label", type=str)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--hold", type=int, default=5)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from stage20_cfg import FrankaLiftStage20Cfg


class StickyGripper(VecEnvWrapper):
    def __init__(self, venv, hold=5):
        super().__init__(venv)
        self.hold = hold
        self.count = np.zeros(venv.num_envs, dtype=np.int64)
        self.held = np.ones(venv.num_envs, dtype=np.float32)

    def step_async(self, actions):
        actions = np.array(actions, copy=True)
        refresh = self.count % self.hold == 0
        self.held = np.where(refresh, actions[:, -1], self.held).astype(np.float32)
        actions[:, -1] = self.held
        self.count += 1
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if np.any(dones):
            self.count[dones] = 0
            self.held[dones] = 1.0
        return obs, rewards, dones, infos

    def reset(self):
        self.count[:] = 0
        self.held[:] = 1.0
        return self.venv.reset()


env_cfg = FrankaLiftStage20Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)

model = PPO.load(args.checkpoint, env=vec)

obs = vec.reset()
prev_action = None
arm_deltas, grip_deltas, joint_speeds, reversals = [], [], [], None
prev2 = None
n_rev_steps = 0

for step in range(args.steps):
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action)

    if prev_action is not None:
        d = np.abs(action - prev_action)
        arm_deltas.append(d[:, :-1].mean(axis=1))   # 7 arm channels
        grip_deltas.append(d[:, -1])                 # gripper channel
        if prev2 is not None:
            # int64, NOT bool: numpy's `+` on bool arrays is a logical OR, so
            # accumulating bools saturates at True and the rate comes out as
            # 1/steps no matter how much the arm actually shakes.
            sign_flip = (np.sign(action[:, :-1] - prev_action[:, :-1]) !=
                         np.sign(prev_action[:, :-1] - prev2[:, :-1])).astype(np.int64)
            reversals = sign_flip if reversals is None else (reversals + sign_flip)
            n_rev_steps += 1
    prev2 = prev_action
    prev_action = action.copy()

    obs, reward, done, info = vec.step(action)
    # arm joints only: the Franka's 7 revolute joints, excluding the two fingers
    joint_speeds.append(
        raw.scene["robot"].data.joint_vel[:, :7].abs().mean(dim=1).cpu().numpy()
    )

arm_deltas = np.concatenate(arm_deltas)
grip_deltas = np.concatenate(grip_deltas)
joint_speeds = np.concatenate(joint_speeds)
rev_rate = float(np.mean(reversals) / n_rev_steps) if n_rev_steps else float("nan")

lines = [
    "SMOOTHNESS PROBE -- %s" % args.label,
    "  checkpoint %s" % args.checkpoint,
    "  %d envs x %d steps, deterministic (no sampling: what shakes is the policy"
    % (args.num_envs, args.steps),
    "  reacting to observations, not exploration noise)",
    "",
    "  arm action delta per step   median %.4f   p95 %.4f   max %.4f"
    % (np.median(arm_deltas), np.percentile(arm_deltas, 95), arm_deltas.max()),
    "  gripper action delta        median %.4f   p95 %.4f"
    % (np.median(grip_deltas), np.percentile(grip_deltas, 95)),
    "  arm joint speed (rad/s)     median %.4f   p95 %.4f   max %.4f"
    % (np.median(joint_speeds), np.percentile(joint_speeds, 95), joint_speeds.max()),
    "  direction reversals per arm channel per step   %.3f" % rev_rate,
    "",
    "  Action channels are in [-1, 1], so an arm delta of 0.05 is a 2.5%% swing of",
    "  the full command range EVERY control step. A reversal rate near 0.5 means",
    "  a channel changes direction on about half of all steps, which is shake",
    "  rather than motion.",
]
out = "smoothness_%s.txt" % args.label
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

vec.close()
simulation_app.close()
