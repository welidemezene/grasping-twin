"""Run a trained checkpoint in a loop and write the current frame to viewer/live.json.

Isaac Sim cannot render on this machine, but it can write. The browser polls this
file ten times a second and draws the arm from it — the same drawing code the
recorded replay uses, just fed live. Stop it with Ctrl+C.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json, os, sys, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

CKPT = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage1_final"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "../viewer/live.json"

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)

model = PPO.load(CKPT, env=vec)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

obs = vec.reset()
step = 0
while simulation_app.is_running():
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)

    frame = {
        "step": step,
        "joints": raw.scene["robot"].data.joint_pos[0].cpu().tolist(),
        "cube":   raw.scene["object"].data.root_pos_w[0].cpu().tolist(),
        "ee":     raw.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist(),
        "base":   raw.scene["robot"].data.root_pos_w[0].cpu().tolist(),
        "done":   bool(done[0]),
    }

    # write to a temp file then rename, so the browser never reads half a file
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(frame, f)
    os.replace(tmp, OUT)

    step += 1

vec.close()
simulation_app.close()
