"""Stage 12 — same rig as stage 11, minus the hidden movement tax.

The TensorBoard curves for stages 10 and 11 are near-identical and both fall
off a cliff at ~5.2M steps: ep_rew 19.6 -> 0.3 in one iteration. That is the
stock lift task's built-in curriculum firing — it multiplies the action_rate
and joint_vel penalty weights by 1000x (-1e-4 -> -1e-1) after 10,000 per-env
steps, which at 512 envs is ~5.1M total steps. Every long run this project
ever did spent its second half under that tax, learning to stop moving —
that, not the grasp reward, is why the 10M finals always froze 2 cm out.

curriculum_lift_cfg now disables both curriculum terms. This script is
otherwise stage 11 unchanged (sticky 5-frame gripper, ent_coef 0.01),
warm-started from stage 11's pre-cliff peak.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage11_4992000_steps")
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--hold", type=int, default=5, help="frames each gripper decision is held")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from curriculum_lift_cfg import FrankaLiftStage1Cfg


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (see stage 11)."""

    def __init__(self, venv, hold=5):
        super().__init__(venv)
        self.hold = hold
        self.count = np.zeros(venv.num_envs, dtype=np.int64)
        self.held = np.ones(venv.num_envs, dtype=np.float32)  # start open

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


env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage12",
    custom_objects={"ent_coef": args.ent_coef},
)

report = []
with torch.no_grad():
    report.append("start_from %s" % args.start_from)
    report.append("stock curriculum disabled: action_rate and joint_vel stay at base weights")
    report.append("gripper bias %.4f  log_std %.3f"
                  % (float(model.policy.action_net.bias[-1]), float(model.policy.log_std[-1])))
    if float(model.policy.log_std[-1]) < -0.7:
        model.policy.log_std[-1] = -0.5
        report.append("gripper log_std raised to -0.5 (was collapsing)")
    report.append("sticky gripper: decisions held for %d frames" % args.hold)
    report.append("ent_coef %.4f" % model.ent_coef)
open("stage12_reset.txt", "w").write("\n".join(report) + "\n")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage12",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage12_final")

env.close()
simulation_app.close()
