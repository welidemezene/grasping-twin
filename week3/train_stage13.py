"""Stage 13 — second continuation: keep converting exploration into skill.

The stage-12 continuation validated the taper thesis: ent_coef 0.01 -> 0.003
sent std falling (0.87 -> 0.69) while ep_rew rose (22.5 -> 26.0), and the
curve was STILL climbing at the 10M cutoff — the budget, not the learning,
was the limiter. Deterministic replays show the grasp condensing out of the
noise: the first in-band GRASP frame (finger sum 0.0540 on the cube) appeared
at the previous final, and closes now reach 0.064 near the cube.

Stage 13 continues from stage12_final with the taper's next step
(ent_coef 0.001) and — lesson learned — its OWN checkpoint prefix: rerunning
train_stage12.py overwrote the first stage-12 series, because SB3's
CheckpointCallback names files by prefix+step with no run identity. Weights
survive inside the lineage, but history should never be overwritten. One
stage, one prefix, from now on.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage12_final")
parser.add_argument("--ent_coef", type=float, default=0.001)
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
    args.start_from, env=env, tensorboard_log="week3/logs/stage13",
    custom_objects={"ent_coef": args.ent_coef},
)

report = []
with torch.no_grad():
    report.append("start_from %s" % args.start_from)
    report.append("stock curriculum disabled: action_rate and joint_vel stay at base weights")
    report.append("gripper bias %.4f  log_std %.3f"
                  % (float(model.policy.action_net.bias[-1]), float(model.policy.log_std[-1])))
    report.append("sticky gripper: decisions held for %d frames" % args.hold)
    report.append("ent_coef %.4f (taper step 3: 0.01 -> 0.003 -> 0.001)" % model.ent_coef)
open("stage13_reset.txt", "w").write("\n".join(report) + "\n")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage13",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage13_final")

env.close()
simulation_app.close()
