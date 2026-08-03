"""Build the env and step it, so a broken reward fails in 2 minutes instead of 42.

Config that parses is not config that runs: wrong param names or tensor shapes in
a reward term only surface when the reward manager actually calls it.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

lines = []
env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 16
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)

obs = vec.reset()
totals = {}
for step in range(30):
    action = torch.zeros((16, raw.action_space.shape[-1])).uniform_(-1, 1).numpy()
    obs, reward, done, info = vec.step(action)
    for name, buf in raw.reward_manager._episode_sums.items():
        totals[name] = float(buf.mean())

lines.append("stepped 30 times with random actions — no crash")
lines.append("")
lines.append("mean episode reward so far, per term:")
for name, val in totals.items():
    lines.append("  %-34s %+.4f" % (name, val))

held = raw.reward_manager._episode_sums.get("lifting_object")
lines.append("")
lines.append("lifting_object is %s — expected ~0 with random actions, since it now needs a real hold"
             % ("ZERO" if abs(float(held.mean())) < 1e-6 else "NONZERO"))

open("smoke_report.txt", "w").write("\n".join(lines) + "\n")

vec.close()
simulation_app.close()
