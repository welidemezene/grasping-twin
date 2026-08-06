"""Render a trained policy with the REAL Isaac Sim renderer, not the hand-drawn viewer.

Why this exists: the browser viewer draws the gripper as approximated boxes, and
at the true finger clearance (~1.5 mm per face) that approximation renders as the
fingers passing THROUGH the cube. It looks like a physics violation and is not
one -- but it is impossible to tell from a hand-drawn picture, and it has now
misled us twice (the 7 cm TCP offset was the first time).

This replays a checkpoint inside Isaac Sim with a camera attached, so what you
see is the actual Franka mesh, the actual cube, actual contact, actual lighting.
If the fingers really did intersect the cube, this is where it would show.

The camera tracks the midpoint between the fingertips and the cube, so the grasp
stays centred and close through the whole episode.

Outputs PNG frames, and an MP4 if ffmpeg is available in the container.

Usage (inside the container, from /workspace/week3):
    render_isaac.py --checkpoint week3/checkpoints/stage16_final --out_dir render_s16
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="week3/checkpoints/stage16_final")
parser.add_argument("--out_dir", type=str, default="render_s16")
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--num_envs", type=int, default=512,
                    help="512 matches the other replays, so the trajectory is "
                         "the same one the measurements describe")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cam_dist", type=float, default=0.45,
                    help="metres back from the grasp point")
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

from isaaclab.app import AppLauncher

# enable_cameras is the whole point -- without it the renderer is not started
# and camera.data.output comes back empty.
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg


class StickyGripper(VecEnvWrapper):
    """The 5-frame gripper hold the policy was trained under (stages 11+)."""

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


os.makedirs(args.out_dir, exist_ok=True)

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

# ONE camera at an absolute prim path, not "{ENV_REGEX_NS}/..." -- the regex form
# would build one camera per environment, i.e. 512 of them.
env_cfg.scene.render_cam = CameraCfg(
    prim_path="/World/RenderCam",
    update_period=0.0,
    width=args.width,
    height=args.height,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
        clipping_range=(0.01, 100.0),
    ),
)

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)
vec = StickyGripper(vec, hold=args.hold)

model = PPO.load(args.checkpoint, env=vec)

cam: Camera = raw.scene["render_cam"]
dev = raw.device


def save_png(arr, path):
    """arr is HxWx3 uint8."""
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return True
    except Exception:
        # Fall back to a raw PPM, which ffmpeg reads happily.
        with open(path.replace(".png", ".ppm"), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (arr.shape[1], arr.shape[0]))
            f.write(arr.tobytes())
        return False


obs = vec.reset()
saved = 0
used_png = True

for step in range(args.steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)

    # Aim at the midpoint of fingertips and cube: that is the grasp, and it is
    # what we are actually here to inspect.
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0, :]
    cube = raw.scene["object"].data.root_pos_w[0]
    target = (ee + cube) * 0.5

    # Sit back, to one side and slightly above -- a three-quarter view shows the
    # gap between finger and cube face, which a head-on view hides.
    d = args.cam_dist
    eye = target + torch.tensor([0.75 * d, -0.85 * d, 0.45 * d], device=dev)

    cam.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    cam.update(dt=0.0)

    rgb = cam.data.output["rgb"]
    if rgb is not None and rgb.shape[0] > 0:
        frame = rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        used_png = save_png(frame, os.path.join(args.out_dir, "frame_%04d.png" % step))
        saved += 1

    if bool(done[0]):
        break

print("[render] saved %d frames to %s" % (saved, args.out_dir))

# Stitch to MP4 if ffmpeg exists. Not fatal if it does not.
import shutil
import subprocess

if shutil.which("ffmpeg") and saved:
    ext = "png" if used_png else "ppm"
    mp4 = os.path.join(args.out_dir, "..", os.path.basename(args.out_dir) + ".mp4")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(args.out_dir, "frame_%%04d.%s" % ext),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        print("[render] wrote %s" % mp4)
    except Exception as exc:
        print("[render] ffmpeg failed (%s) -- PNGs are still there" % type(exc).__name__)
else:
    print("[render] no ffmpeg in the container; PNG frames written, stitch them outside")

vec.close()
simulation_app.close()
