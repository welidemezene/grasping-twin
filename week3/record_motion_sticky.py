"""STICKY replay: same as record_motion.py but reproduces the 5-frame gripper
hold the policy was TRAINED under (stages 11-13). record_motion.py steps the raw
env, so a close command that lasted 5 frames in training lasts 1 in replay --
and finger_report.txt says fingers need ~4-5 consecutive close frames to reach
the cube. This script tests whether the "no grasp" verdicts are a replay
artifact rather than a policy failure.

Original: Replay a trained checkpoint and save joint angles + cube/gripper/base positions to JSON.

Records the true end-effector frame (same source find_gripper.py uses), the robot
base pose, and the episode-done flag, so gripper-to-cube distance is measured
rather than assumed. Frames after the first episode reset are dropped — the cube
snapping back to its spawn height belongs to the next episode, not this replay.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json, sys, torch, numpy as np, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (mirrors train_stage13.py)."""

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


CKPT = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage1_final"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "motion.json"

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)
vec = StickyGripper(vec, hold=int(sys.argv[3]) if len(sys.argv) > 3 else 5)

model = PPO.load(CKPT, env=vec)

frames = []
obs = vec.reset()
for step in range(250):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frames.append({
        "step": step,
        "joints": raw.scene["robot"].data.joint_pos[0].cpu().tolist(),
        "cube":   raw.scene["object"].data.root_pos_w[0].cpu().tolist(),
        "ee":     raw.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist(),
        "base":   raw.scene["robot"].data.root_pos_w[0].cpu().tolist(),
        "done":   bool(done[0]),
    })
    if bool(done[0]):
        break

# drop the reset frame itself: its cube position is the next episode's spawn
if frames and frames[-1]["done"]:
    frames.pop()

with open(OUT, "w") as f:
    json.dump(frames, f)

with open(OUT.replace(".json", "_summary.txt"), "w") as f:
    z = [fr["cube"][2] for fr in frames]
    gap = [sum((fr["ee"][i] - fr["cube"][i]) ** 2 for i in range(3)) ** 0.5 for fr in frames]
    fsum = [fr["joints"][7] + fr["joints"][8] for fr in frames]
    # a grasp is fingers stopped BY the cube (sum near 0.042), not a shut
    # fist (~0.004) — the old `< 0.005` test could never see a successful
    # grasp, only an empty fist
    grasp_frames = [i for i in range(len(frames))
                    if abs(fsum[i] - 0.042) < 0.012 and gap[i] < 0.03]
    fist = next((i for i, s in enumerate(fsum) if s < 0.01), None)
    f.write(f"checkpoint {CKPT}\nframes {len(frames)}\n")
    f.write(f"cube z: start {z[0]:.4f} min {min(z):.4f} max {max(z):.4f} end {z[-1]:.4f}\n")
    f.write(f"gripper-cube distance: start {gap[0]:.4f} min {min(gap):.4f} (frame {gap.index(min(gap))}) end {gap[-1]:.4f}\n")
    if grasp_frames:
        g0 = grasp_frames[0]
        f.write(f"GRASP: fingers stopped on the cube for {len(grasp_frames)} frames, "
                f"first at frame {g0} (finger sum {fsum[g0]:.4f}, gap {gap[g0]:.4f})\n")
        lifted = [i for i in grasp_frames if z[i] > 0.035]
        if lifted:
            f.write(f"HELD LIFT: cube above 0.035 while grasped for {len(lifted)} frames, "
                    f"first at frame {lifted[0]} (cube z {z[lifted[0]]:.4f})\n")
        else:
            f.write("no held lift yet: grasped but cube never above 0.035\n")
    else:
        i = fsum.index(min(fsum))
        f.write(f"no grasp: min finger sum {fsum[i]:.4f} at frame {i} (gap there {gap[i]:.4f})\n")
    if fist is not None:
        f.write(f"note: empty fist (sum < 0.01) at frame {fist}, gap {gap[fist]:.4f} — closing on nothing\n")

vec.close()
simulation_app.close()
