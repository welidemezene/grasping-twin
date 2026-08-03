"""Ask the policy why it never closes the gripper.

The reward pays clearly more for closing at the cube, so if the fingers never
shut, either the policy cannot express it or it never samples it. This prints
the exploration spread per action dimension and what the gripper dimension
actually outputs while the gripper is sitting on the cube.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import sys, torch, numpy as np, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

CKPT = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage6_1996800_steps"

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 64
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)
model = PPO.load(CKPT, env=vec)

lines = [f"checkpoint {CKPT}"]

log_std = model.policy.log_std.detach().cpu().numpy()
lines.append("action dims: %d   (last one is the gripper)" % len(log_std))
lines.append("log_std per dim: " + " ".join("%.2f" % v for v in log_std))
lines.append("std     per dim: " + " ".join("%.3f" % v for v in np.exp(log_std)))
lines.append("")

obs = vec.reset()
grip_mean, grip_sampled, dists = [], [], []
for step in range(250):
    act_det, _ = model.predict(obs, deterministic=True)
    act_rnd, _ = model.predict(obs, deterministic=False)
    grip_mean.append(float(act_det[0, -1]))
    grip_sampled.append(float(act_rnd[0, -1]))
    cube = raw.scene["object"].data.root_pos_w[0]
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0, :]
    dists.append(float(torch.norm(cube - ee)))
    obs, reward, done, info = vec.step(act_det)
    if bool(done[0]):
        break

gm = np.array(grip_mean); gs = np.array(grip_sampled); dd = np.array(dists)
lines.append("gripper action, deterministic (what the replay uses):")
lines.append("  min %.3f  max %.3f  mean %.3f   frames below 0 (= close command): %d / %d"
             % (gm.min(), gm.max(), gm.mean(), int((gm < 0).sum()), len(gm)))
lines.append("gripper action, sampled (what training explores with):")
lines.append("  min %.3f  max %.3f  mean %.3f   frames below 0: %d / %d"
             % (gs.min(), gs.max(), gs.mean(), int((gs < 0).sum()), len(gs)))
near = dd < 0.03
if near.any():
    lines.append("")
    lines.append("while INSIDE 3 cm (%d frames): deterministic gripper mean %.3f, min %.3f"
                 % (int(near.sum()), gm[near].mean(), gm[near].min()))
    lines.append("                                sampled       gripper min  %.3f, frames<0 %d"
                 % (gs[near].min(), int((gs[near] < 0).sum())))

open("policy_report.txt", "w").write("\n".join(lines) + "\n")

vec.close()
simulation_app.close()
