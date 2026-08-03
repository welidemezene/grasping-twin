"""Measure gripper and cube positions at reset. Writes to a file (prints get lost in Docker)."""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped

obs, _ = env.reset()
for _ in range(5):
    env.step(torch.zeros(raw.num_envs, *env.action_space.shape[1:], device=raw.device))

ee   = raw.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist()
cube = raw.scene["object"].data.root_pos_w[0].cpu().tolist()

out = {
    "gripper": ee,
    "cube": cube,
    "offset_to_gripper": [ee[0]-cube[0], ee[1]-cube[1], ee[2]-cube[2]],
}
with open("gripper_pos.json", "w") as f:
    json.dump(out, f, indent=2)

env.close()
simulation_app.close()
