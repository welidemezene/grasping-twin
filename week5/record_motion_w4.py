"""Export ONE arm's trajectory to JSON so the browser viewer can replay it.

week3/record_motion_sticky.py does this for week 3, but it hardcodes
FrankaLiftStage1Cfg, so it cannot record any week 4 policy. This is the week 4
version, and it changes one thing that matters:

    IT RECORDS AT A FIXED CUBE OFFSET, NOT A RANDOM ONE.

That is the whole point of the viewer. FrankaLiftStage20Cfg spawns the cube
anywhere in a +-5 cm box, so two policies recorded under it see two different
cubes and the replays cannot be compared -- the same mistake as ranking policies
off a 9-episode render. FrankaLiftShiftedCfg puts every env at the SAME offset,
which is also how eval_shift.py measured the numbers, so a pair of recordings at
one offset is directly comparable and the sweep row applies to both.

Default offset is the 5 cm diagonal (x +0.035, y +0.035), chosen from the sweep
rather than for looks -- it is where the gap between a policy that memorised a
trajectory and one that reads the cube is widest and still honest:

    offset            stage 16   stage 22
    +x 4 cm             41.8%      87.7%
    5 cm diagonal       41.0%      93.2%     <- widest honest gap
    7 cm diagonal        0.0%      62.7%     <- both drop cubes here

STICKY GRIPPER IS NOT OPTIONAL. Every policy from stage 11 on trained with the
gripper channel held 5 frames per decision. Step the raw env instead and a close
command that lasted 5 frames in training lasts 1 in replay, the fingers never
reach the cube, and the replay shows a failure the policy does not have. Week 3
lost real time to exactly this.

OUTPUT FORMAT IS AN OBJECT, NOT A BARE ARRAY. week 3's files are a bare list of
frames, which leaves the viewer with no way to state which checkpoint or offset
it is showing -- so a caption has to be typed by hand, and a hand-typed number
is a number that can drift from the file it claims to come from. This writes
{"meta": {...}, "frames": [...]}. The viewer must accept both shapes.

Derived per-frame flags (grasped, lifted) are computed HERE, with the same
geometry the week 3 summaries use, so the browser never reimplements grasp
detection in JavaScript. A grasp is fingers stopped BY the cube (finger sum near
0.042 while the gripper is within 3 cm), never a shut fist -- an empty fist sums
to about 0.004 and the old `< 0.005` test could only ever see that.

Usage (Docker, from WSL -- the week 4 pattern, one container per recording
because Isaac Lab's simulation context is a singleton and hangs if a second
scene is built in the same process):

    docker run --rm --gpus all -v ~/grasping_twin:/workspace \
        -w /workspace/week4 grasping-twin-isaaclab:latest \
        ../week5/record_motion_w4.py checkpoints/stage22_final \
        ../week5/motion/s22_diag50.json --shift_x 0.035 --shift_y 0.035

Vulkan errors in the log are always present in headless mode and are not a
signal. Two week-3 launch failures looked like config problems and were both
transient: retry once before debugging.
"""
import argparse
import json
import os
import sys

# week4/ holds stage20_cfg, which puts week3/ on the path itself. Anchor to this
# file rather than the working directory, so the script runs from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week4"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ckpt", help="checkpoint path, no .zip")
parser.add_argument("out", help="output .json path")
parser.add_argument("--shift_x", type=float, default=0.035)
parser.add_argument("--shift_y", type=float, default=0.035)
parser.add_argument("--hold", type=int, default=5, help="sticky gripper frames; 5 is what training used")
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--envs", type=int, default=1,
                    help="every env gets the SAME offset here, so 1 is enough; raise it if the scene misbehaves")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--label", default=None, help="short name for the viewer, e.g. 'stage 22'")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402

from stage20_cfg import FrankaLiftShiftedCfg  # noqa: E402


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (mirrors training)."""

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


env_cfg = FrankaLiftShiftedCfg()
env_cfg.shift_x = args.shift_x
env_cfg.shift_y = args.shift_y
env_cfg.__post_init__()          # shift_x/shift_y are read there, so re-run it
env_cfg.scene.num_envs = args.envs
env_cfg.seed = args.seed

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)

model = PPO.load(args.ckpt, env=vec)

frames = []
obs = vec.reset()
for step in range(args.steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frames.append({
        "step": step,
        "joints": raw.scene["robot"].data.joint_pos[0].cpu().tolist(),
        "cube": raw.scene["object"].data.root_pos_w[0].cpu().tolist(),
        "ee": raw.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist(),
        "base": raw.scene["robot"].data.root_pos_w[0].cpu().tolist(),
        "done": bool(done[0]),
    })
    if bool(done[0]):
        break

# The reset frame's cube position belongs to the NEXT episode -- the cube snaps
# back to spawn height and a viewer reads it as a dropped cube.
if frames and frames[-1]["done"]:
    frames.pop()

if not frames:
    print("NO FRAMES RECORDED -- refusing to write a file", file=sys.stderr)
    vec.close()
    simulation_app.close()
    sys.exit(1)

# Derived flags live in derive_flags.py, so the recorder, the standalone
# re-derivation pass, and the browser cannot drift apart on what a grasp is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_flags import annotate, summary_text  # noqa: E402

episode = annotate(frames)

meta = {
    "checkpoint": args.ckpt,
    "label": args.label or os.path.basename(args.ckpt),
    "shift_x": args.shift_x,
    "shift_y": args.shift_y,
    "shift_diagonal_m": round((args.shift_x ** 2 + args.shift_y ** 2) ** 0.5, 4),
    "hold": args.hold,
    "seed": args.seed,
    "envs": args.envs,
    "frames": len(frames),
    "config": "FrankaLiftShiftedCfg",
    # Outcome of THIS ONE episode -- an illustration, never a success rate.
    # The verdict at this offset lives in week4/*_sweep.csv, 512 trials.
    "episode": episode,
}

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"meta": meta, "frames": frames}, f)

summary = args.out.replace(".json", "_summary.txt")
with open(summary, "w") as f:
    f.write(summary_text(meta))

print(f"wrote {args.out}  ({len(frames)} frames)")
print(open(summary).read())

vec.close()
simulation_app.close()
