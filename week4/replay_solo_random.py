"""One arm, many episodes, a different cube position every time. The social cut.

The grid videos fail as a public artefact for opposite reasons, and this fixes
both:

  - stage22_grid25.mp4 randomizes the spawn but shows 25 arms at once, so a
    +-5 cm shift on a 4.2 cm cube is about two cube-widths in a wide frame --
    invisible. A viewer sees 25 identical robots and reasonably assumes a
    replayed animation.
  - shift50_s*_grid25.mp4 shows the failure honestly, but it is a comparison of
    two policies at ONE offset. It proves the point to someone reading the
    numbers; it does not show a robot handling variety.

A solo close-up inverts the ratio. The camera is on one arm, so +-5 cm is a
large fraction of the frame, and consecutive episodes put the cube somewhere
visibly different each time. Same policy, no cuts, no resets hidden.

FILMED STRAIGHT THROUGH THE RESETS, as week 3's 42-second cut was. The episode
boundary is visible and that is the point: the viewer watches the cube jump to a
new spot and the same arm go get it. Hiding the reset would make it look edited,
which is exactly the suspicion the footage exists to answer.

Per episode this records where the cube spawned and whether it was lifted, then
prints the table. A caption should be able to say "six positions, six lifts, one
policy" and have a file backing every word. The grasp and airborne definitions
are eval_shift.py's, so an episode counted here means the same thing it means in
the sweeps.

This is an ILLUSTRATION. A handful of episodes cannot restate s22_sweep.csv's 512
trials, and the printed table is there so the clip can never be described as
something the numbers do not support.

Usage, Windows, from the Isaac Lab 2.1 venv (isaaclab.bat -p does NOT resolve
isaaclab from a non-interactive shell -- call the venv python directly, and set
OMNI_KIT_ACCEPT_EULA=YES or Kit blocks on the prompt and dies with EOFError):

    C:\\isaac\\venv310\\Scripts\\python.exe C:\\isaac\\replay_solo_random.py ^
        <ckpt> <out.mp4> <hold> <steps> <seed>
"""
import sys
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import gymnasium as gym
import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

sys.path.insert(0, r"C:\isaac")
from stage20_cfg import FrankaLiftStage20Cfg, SPAWN_RANGE

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"C:\isaac\stage22_final.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\isaac\stage22_solo_random.mp4"
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 5
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1800
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0

HALF = 0.0210        # cube half-width, as eval_shift.py
AIRBORNE = 0.005


class StickyGripper(VecEnvWrapper):
    """The 5-frame gripper hold the policy trained under. Rendering without it
    would show a different action regime than the one every number describes --
    the mismatch that silently invalidated the stage 11-13 verdicts."""

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
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cuda:0"
env_cfg.seed = SEED

# Solo hero framing: close enough that a +-5 cm spawn shift is a large fraction
# of the frame. That is the entire reason this shot exists.
env_cfg.viewer.eye = (1.4, -1.2, 0.9)
env_cfg.viewer.lookat = (0.45, 0.0, 0.25)

# Debug visualisations off: goal-pose arrows and ee-frame markers read as UI
# clutter in footage and imply the robot is being told where to go.
try:
    env_cfg.commands.object_pose.debug_vis = False
except AttributeError:
    pass
try:
    env_cfg.scene.ee_frame.debug_vis = False
except AttributeError:
    pass

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=HOLD)

model = PPO.load(CKPT, env=vec)
print("PPO loaded OK: %s" % CKPT, flush=True)
print("spawn range +-%.3f m, one arm, filmed through the resets" % SPAWN_RANGE, flush=True)

writer = imageio.get_writer(OUT, fps=30, quality=8)
obs = vec.reset()
origin = raw.scene.env_origins[0, :2].clone()


def cube_xy():
    return (raw.scene["object"].data.root_pos_w[0, :2] - origin).cpu().numpy()


episodes = []            # (spawn_x, spawn_y, grasped, lifted, frames)
cur_spawn = cube_xy()
cur_grasped = False
cur_lifted = False
cur_frames = 0

for step in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frame = env.render()
    if frame is not None:
        writer.append_data(np.asarray(frame))
    cur_frames += 1

    cube = raw.scene["object"]
    robot = raw.scene["robot"]
    eef = raw.scene["ee_frame"]
    pos = cube.data.root_pos_w
    quat = cube.data.root_quat_w
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    drop = (r20.abs() + r21.abs() + r22.abs()) * HALF
    lowest = float((pos[0, 2] - drop[0]).cpu())
    fsum = float(robot.data.joint_pos[0, -2:].sum().cpu())
    gap = float(torch.norm(pos[0] - eef.data.target_pos_w[0, 0, :]).cpu())

    g = abs(fsum - 0.042) < 0.012 and gap < 0.03
    cur_grasped |= g
    cur_lifted |= (g and lowest > AIRBORNE)

    if bool(done[0]):
        episodes.append((cur_spawn[0], cur_spawn[1], cur_grasped, cur_lifted, cur_frames))
        # the reset has already happened, so this reads the NEXT episode's cube
        cur_spawn = cube_xy()
        cur_grasped = cur_lifted = False
        cur_frames = 0

writer.close()

lifted = sum(1 for e in episodes if e[3])
lines = [
    "SOLO RANDOM-SPAWN CUT -- %s" % CKPT,
    "  %s, %d frames at 30 fps = %.1f s" % (OUT, STEPS, STEPS / 30.0),
    "  one arm, spawn randomized +-%.3f m, filmed through the resets" % SPAWN_RANGE,
    "",
    "  ep   spawn x    spawn y   grasped  lifted   frames",
]
for i, (sx, sy, g, l, f) in enumerate(episodes):
    lines.append("  %2d   %+.4f   %+.4f      %s      %s     %4d"
                 % (i + 1, sx, sy, "yes" if g else "NO ", "yes" if l else "NO ", f))
lines += [
    "",
    "  %d complete episodes, %d lifted" % (len(episodes), lifted),
]
if episodes:
    xs = [e[0] for e in episodes]
    ys = [e[1] for e in episodes]
    lines += [
        "  spawn spread: x %.4f..%.4f (%.1f cm)   y %.4f..%.4f (%.1f cm)"
        % (min(xs), max(xs), (max(xs) - min(xs)) * 100,
           min(ys), max(ys), (max(ys) - min(ys)) * 100),
        "  cube is 4.2 cm wide -- compare the spread to that when judging the shot",
    ]
lines += [
    "",
    "  Illustration, not evidence: a handful of episodes cannot restate",
    "  s22_sweep.csv's 512 trials. Caption from this table, not from the video.",
]
out_txt = OUT.rsplit(".", 1)[0] + "_episodes.txt"
open(out_txt, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

vec.close()
app.close()
