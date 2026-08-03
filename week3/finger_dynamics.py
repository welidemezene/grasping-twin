"""How many steps does a close command need before the fingers are actually shut?

The policy samples a close about 12 times per episode at the cube and never
learns it. If the fingers need several consecutive close commands to reach the
threshold the reward tests, then single-step exploration can never be paid for
trying — the chance of sampling close N times in a row is vanishingly small.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 16
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)

n_act = raw.action_space.shape[-1]
obs = vec.reset()

lines = []

# hold the arm still, command CLOSE continuously, watch the fingers move
act = np.zeros((16, n_act), dtype=np.float32)
act[:, -1] = -1.0
trace = []
for step in range(25):
    obs, r, d, i = vec.step(act)
    trace.append(float(raw.scene["robot"].data.joint_pos[0, -2:].sum()))

lines.append("finger SUM while CLOSE is commanded every step (open = 0.080):")
lines.append("  " + "  ".join("s%d %.4f" % (i, v) for i, v in enumerate(trace[:14])))
below = next((i for i, v in enumerate(trace) if v < 0.06), None)
lines.append("  steps to cross the 0.06 threshold the reward tests: %s" % below)

# now the realistic case: ONE close command, then back to open
obs = vec.reset()
act_open = np.zeros((16, n_act), dtype=np.float32); act_open[:, -1] = 1.0
act_shut = np.zeros((16, n_act), dtype=np.float32); act_shut[:, -1] = -1.0
for _ in range(5):
    obs, r, d, i = vec.step(act_open)
one = [float(raw.scene["robot"].data.joint_pos[0, -2:].sum())]
obs, r, d, i = vec.step(act_shut)
one.append(float(raw.scene["robot"].data.joint_pos[0, -2:].sum()))
for _ in range(4):
    obs, r, d, i = vec.step(act_open)
    one.append(float(raw.scene["robot"].data.joint_pos[0, -2:].sum()))

lines.append("")
lines.append("finger SUM for a SINGLE close command then open again:")
lines.append("  " + "  ".join("%.4f" % v for v in one))
lines.append("  lowest reached: %.4f  (threshold is 0.060)" % min(one))

open("finger_report.txt", "w").write("\n".join(lines) + "\n")

vec.close()
simulation_app.close()
