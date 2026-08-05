"""Diagnose WHY the fingers stop at sum ~0.060 while parked <1 mm from the cube.

Nine stages (5-13) converged on "approach, don't close". That survived two
gripper-head resets, a reward rewrite, the sticky gripper and the curriculum
fix, so it is structural. This probe measures the three things that separate
the surviving explanations, on the frames where it matters (gap < 5 mm):

  1. THE COMMAND. The gripper channel is binarised downstream (open/close), so
     the question is not "how hard" but "how often": is close actually being
     commanded, and for how many consecutive frames? finger_report.txt says a
     sustained close reaches 0.0550 by step 2 and 0.0386 by step 4. Under
     hold=5 a genuine close command MUST reach 0.0386. Stopping at 0.0602 means
     either the command is not sustained, or something physically blocks it.

  2. THE ORIENTATION -- the measurement this project has never taken.
     record_motion.py records no quaternions. A 4.2 cm cube gripped face-to-face
     stops the fingers at 0.042; gripped across its DIAGONAL it stops them at
     0.042*sqrt(2) = 0.0594. The observed plateau is 0.0602. If the gripper is
     yawed ~45 deg to the cube, the fingers are already on the cube, closing is
     physically blocked, and every "no grasp" verdict has been a ruler built for
     the wrong approach angle. yaw_to_cube_deg below settles it.

  3. THE REWARD. Per-term, per-frame. Two things to read: does _held ever fire
     (its gate is fsum < 0.061, LOOSER than the recorder's < 0.054 -- so it may
     already be firing while replays say "no grasp"), and is there still gradient
     left for closing further, or has fingers_on_cube gone flat?

Writes a per-frame CSV plus a verdict summary that names the surviving cause.

Usage:
  python week3/probe_grasp.py <checkpoint> [out_prefix] [hold]
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import csv, math, sys, torch, numpy as np, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg, HOLD_DISTANCE
import grasp_reward
from grasp_reward import GRASP_SUM, OPEN_SUM

NEAR = 0.005          # "at the cube" for the purposes of this probe
DIAGONAL = GRASP_SUM * math.sqrt(2.0)


class StickyGripper(VecEnvWrapper):
    """Same wrapper training used, but it also exposes what it actually sent."""

    def __init__(self, venv, hold=5):
        super().__init__(venv)
        self.hold = hold
        self.count = np.zeros(venv.num_envs, dtype=np.int64)
        self.held = np.ones(venv.num_envs, dtype=np.float32)
        self.last_raw = np.zeros(venv.num_envs, dtype=np.float32)
        self.last_refresh = np.zeros(venv.num_envs, dtype=bool)

    def step_async(self, actions):
        actions = np.array(actions, copy=True)
        refresh = self.count % self.hold == 0
        self.last_raw = actions[:, -1].astype(np.float32).copy()
        self.last_refresh = refresh.copy()
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


def yaw_of(quat):
    """Yaw (deg) about z from a (w,x,y,z) quaternion."""
    w, x, y, z = [float(v) for v in quat]
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


CKPT   = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage13_final"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "probe_s13"
HOLD   = int(sys.argv[3]) if len(sys.argv) > 3 else 5

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=HOLD)
model = PPO.load(CKPT, env=vec)

rows = []
obs = vec.reset()
for step in range(250):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)

    robot, cube, eef = raw.scene["robot"], raw.scene["object"], raw.scene["ee_frame"]
    fsum = float(robot.data.joint_pos[0, -2:].sum())
    ee_pos = eef.data.target_pos_w[0, 0, :]
    gap = float(torch.norm(cube.data.root_pos_w[0] - ee_pos))

    # channel 3: per-term reward, computed exactly as the cfg wires it
    approach = float(grasp_reward.approach_object(raw)[0])
    on_cube  = float(grasp_reward.fingers_on_cube(raw, in_position_std=0.015)[0])
    held     = bool(grasp_reward._held(raw, HOLD_DISTANCE,
                                       grasp_reward.SceneEntityCfg("object"),
                                       grasp_reward.SceneEntityCfg("ee_frame"),
                                       grasp_reward.SceneEntityCfg("robot"))[0])
    bump = float(grasp_reward._on_cube_bump(torch.tensor([fsum]))[0])

    # channel 2: the orientation nobody has measured
    cube_yaw = yaw_of(cube.data.root_quat_w[0])
    ee_yaw   = yaw_of(eef.data.target_quat_w[0, 0, :])
    # cube faces repeat every 90 deg, so fold into [0, 45]: 0 = square on a
    # face, 45 = square on a corner (the diagonal case)
    yaw_to_cube = abs((ee_yaw - cube_yaw + 45.0) % 90.0 - 45.0)

    rows.append({
        "step": step,
        "gap": round(gap, 5),
        "finger_sum": round(fsum, 5),
        "raw_grip_action": round(float(vec.last_raw[0]), 4),
        "sent_grip_action": round(float(vec.held[0]), 4),
        "refresh": int(vec.last_refresh[0]),
        "commands_close": int(vec.held[0] <= 0.0),
        "yaw_to_cube_deg": round(yaw_to_cube, 2),
        "cube_z": round(float(cube.data.root_pos_w[0, 2]), 5),
        "r_approach": round(approach * 4.0, 4),
        "r_fingers_on_cube": round(on_cube * 6.0, 4),
        "bump": round(bump, 4),
        "held": int(held),
        "reward_total": round(float(reward[0]), 4),
    })
    if bool(done[0]):
        rows.pop()
        break

with open(f"{PREFIX}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

near = [r for r in rows if r["gap"] < NEAR]
out = []
out.append(f"checkpoint {CKPT}   hold {HOLD}   frames {len(rows)}   near-frames (gap<{NEAR}) {len(near)}")
out.append(f"grasp band: face-on stop {GRASP_SUM:.4f} | diagonal stop {DIAGONAL:.4f} | open {OPEN_SUM:.4f}")
out.append(f"reward _held gate: finger sum < 0.0610   |   replay grasp test: finger sum < 0.0540")

if not near:
    out.append(f"\nNEVER GOT NEAR: min gap {min(r['gap'] for r in rows):.5f}. The arm is the problem, not the hand.")
else:
    closes = sum(r["commands_close"] for r in near)
    runs, cur = [], 0
    for r in rows:
        cur = cur + 1 if r["commands_close"] else 0
        runs.append(cur)
    best_run = max(runs)
    fmin = min(r["finger_sum"] for r in near)
    yaws = [r["yaw_to_cube_deg"] for r in near]
    yaw_mean = sum(yaws) / len(yaws)
    held_frames = sum(r["held"] for r in near)
    bumps = [r["bump"] for r in near]

    out.append(f"\n1. COMMAND: close commanded on {closes}/{len(near)} near-frames; "
               f"longest consecutive close run anywhere: {best_run} frames")
    out.append(f"   min finger sum near the cube: {fmin:.4f}")
    out.append(f"2. ORIENTATION: yaw to nearest cube face {min(yaws):.1f}-{max(yaws):.1f} deg "
               f"(mean {yaw_mean:.1f}); 0 = face-on, 45 = corner-on")
    out.append(f"3. REWARD: _held fired on {held_frames}/{len(near)} near-frames; "
               f"bump {min(bumps):.3f}-{max(bumps):.3f}; "
               f"fingers_on_cube {min(r['r_fingers_on_cube'] for r in near):.2f}"
               f"-{max(r['r_fingers_on_cube'] for r in near):.2f} of a possible 6.00")

    out.append("\nVERDICT")
    if closes == 0:
        out.append("  NOT COMMANDING CLOSE. The policy never asks. This is a policy/reward")
        out.append("  problem, not physics -- closing is simply not the action it prefers.")
    elif best_run < 2:
        out.append("  COMMANDING, BUT NOT SUSTAINED. Close is asked for but flips back before")
        out.append("  the fingers travel (needs ~2 frames for 0.055, ~4 for 0.039). Raise the")
        out.append("  sticky hold, or penalise gripper flip-flop.")
    elif yaw_mean > 25.0 and abs(fmin - DIAGONAL) < 0.004:
        out.append(f"  ALREADY GRASPING, ACROSS THE CORNERS. Fingers stop at {fmin:.4f}, which is")
        out.append(f"  the {DIAGONAL:.4f} diagonal of the cube, and the gripper is yawed {yaw_mean:.0f} deg")
        out.append("  off the faces. Closing is PHYSICALLY BLOCKED -- no amount of training will")
        out.append("  close it further. Fix the measurement (accept the diagonal) and add a yaw")
        out.append("  alignment term so the gripper squares up to a face. Do NOT retrain as-is.")
    elif fmin > 0.061:
        out.append("  BLOCKED WELL SHORT OF THE CUBE. Fingers stop wider than even the reward's")
        out.append("  own _held gate. Check for a collision (finger vs cube top/side) or a joint")
        out.append("  limit -- inspect cube_z in the CSV for a nudge at contact.")
    else:
        out.append("  COMMANDING AND SUSTAINED, STOPS IN THE DISPUTED ZONE (0.054-0.061). The")
        out.append("  reward calls this held; the replay does not. Decide which ruler is right")
        out.append("  before another stage -- see yaw_to_cube_deg above for whether contact is real.")

    nudge = max(r["cube_z"] for r in near) - min(r["cube_z"] for r in near)
    out.append(f"\n  cube z moved {nudge:.4f} m across the near-frames "
               f"({'nudged at contact' if nudge > 0.003 else 'stable, not batted away'})")

text = "\n".join(out)
with open(f"{PREFIX}_summary.txt", "w") as f:
    f.write(text + "\n")
print(text)

vec.close()
simulation_app.close()
