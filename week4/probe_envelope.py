"""How far can the cube go before the policy loses it -- and is the wall policy or kinematics?

The table in the footage is far bigger than the +-5 cm patch the cube spawns in.
The question is what it would take to cover it. Two possible walls, and they have
completely different answers:

  POLICY WALL   the arm can reach there, it was never trained to. Fixed by
                widening the spawn distribution and retraining. Cheap.
  KINEMATIC WALL  the arm physically cannot reach there from where it is bolted.
                No amount of RL fixes it; it needs a different mount, a smaller
                table, or an accepted bound. Retraining would burn 45 minutes to
                learn nothing.

sweep_shift.sh answers this one offset per container at ~2 minutes each, so a
grid out to the table edge is dozens of runs. This does it in ONE: spawn the cube
uniformly over a WIDE box, 512 envs, and record for every env where its cube
landed, whether it got lifted, and whether any arm joint sat at its limit. 512
scattered samples ARE the grid.

Reports an ASCII heat map binned by position, so the shape of the envelope is
visible rather than inferred from a list of numbers. Directional brittleness was
the surprise of this project once already -- +x 4 cm scored 41.8% while -y 4 cm
scored 99.6% -- and a radial summary would have hidden it completely.

The grasp/airborne definitions are eval_shift.py's, so a success here means what
it means in the sweeps. Joint saturation uses diagnose_reach.py's test: the
fraction of a joint's soft range, pinned if under 5% or over 95%.

    probe_envelope.py <checkpoint> <label> [--range 0.15]

NOT a substitute for sweep_shift.sh. The sweep puts all 512 envs at ONE offset
and so scores that offset with 512 trials. This spreads 512 trials over the whole
box, so each bin holds tens of samples, not hundreds. It is a map for deciding
where to aim, and the verdict on any specific offset still comes from the sweep.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("label", type=str)
parser.add_argument("--range", type=float, default=0.15,
                    help="half-width of the spawn box in metres (training was 0.05)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--bins", type=int, default=7)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from stage20_cfg import FrankaLiftStage20Cfg

HALF = 0.0210
AIRBORNE = 0.005


class StickyGripper(VecEnvWrapper):
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
env_cfg.scene.num_envs = args.num_envs
# The whole point: spawn far wider than training, to find the wall.
env_cfg.events.reset_object_position.params["pose_range"] = {
    "x": (-args.range, args.range),
    "y": (-args.range, args.range),
    "z": (0.0, 0.0),
}

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)
model = PPO.load(args.checkpoint, env=vec)

N = raw.num_envs
robot = raw.scene["robot"]
arm_lo = robot.data.soft_joint_pos_limits[0, :7, 0]
arm_hi = robot.data.soft_joint_pos_limits[0, :7, 1]

obs = vec.reset()

# Spawn positions are read once, right after reset, BEFORE the arm can move the
# cube. Reading them later would record where the cube ended up, which is a
# different question and would quietly turn the map into nonsense.
origins = raw.scene.env_origins[:, :2]
spawn = (raw.scene["object"].data.root_pos_w[:, :2] - origins).cpu().numpy().copy()

ever_both = torch.zeros(N, dtype=torch.bool)
max_pinned = torch.zeros(N)
alive = torch.ones(N, dtype=torch.bool)

for step in range(args.steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, rew, done, info = vec.step(a)

    cube = raw.scene["object"]
    eef = raw.scene["ee_frame"]
    pos = cube.data.root_pos_w
    quat = cube.data.root_quat_w
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    drop = (r20.abs() + r21.abs() + r22.abs()) * HALF
    lowest = (pos[:, 2] - drop).cpu()

    fsum = robot.data.joint_pos[:, -2:].sum(dim=1).cpu()
    gap = torch.norm(pos - eef.data.target_pos_w[:, 0, :], dim=1).cpu()
    g = ((fsum - 0.042).abs() < 0.012) & (gap < 0.03)
    ever_both |= (g & (lowest > AIRBORNE) & alive)

    frac = (robot.data.joint_pos[:, :7] - arm_lo) / (arm_hi - arm_lo)
    pinned = ((frac < 0.05) | (frac > 0.95)).float().sum(dim=1).cpu()
    max_pinned = torch.maximum(max_pinned, torch.where(alive, pinned, torch.zeros(N)))

    # Freeze at first reset: after that an env's cube is somewhere new and the
    # spawn position recorded above no longer describes it.
    alive &= ~torch.as_tensor(np.asarray(done, dtype=bool))

vec.close()

ok = ever_both.numpy()
pin = max_pinned.numpy()
R = args.range
B = args.bins
edges = np.linspace(-R, R, B + 1)
xi = np.clip(np.digitize(spawn[:, 0] - spawn[:, 0].mean(), edges) - 1, 0, B - 1)
yi = np.clip(np.digitize(spawn[:, 1] - spawn[:, 1].mean(), edges) - 1, 0, B - 1)

grid_n = np.zeros((B, B), dtype=int)
grid_ok = np.zeros((B, B), dtype=int)
grid_pin = np.zeros((B, B))
for i in range(len(ok)):
    grid_n[yi[i], xi[i]] += 1
    grid_ok[yi[i], xi[i]] += int(ok[i])
    grid_pin[yi[i], xi[i]] += pin[i]

lines = [
    "ENVELOPE PROBE -- %s" % args.label,
    "  checkpoint %s" % args.checkpoint,
    "  %d envs, spawn uniform in +-%.3f m (training was +-0.050), first episode only"
    % (N, R),
    "",
    "  overall lifted: %d / %d  (%.1f%%)" % (int(ok.sum()), len(ok), 100.0 * ok.mean()),
    "",
    "  SUCCESS %% BY SPAWN POSITION   (rows = y, +y at top; cols = x, +x at right)",
    "  '.' = no samples in that bin",
    "",
]
head = "        " + "".join("%7.3f" % ((edges[c] + edges[c + 1]) / 2) for c in range(B))
lines.append(head)
for r in range(B - 1, -1, -1):
    row = "  %+.3f " % ((edges[r] + edges[r + 1]) / 2)
    for c in range(B):
        row += ("%6.0f%%" % (100.0 * grid_ok[r, c] / grid_n[r, c])) if grid_n[r, c] else "      ."
    lines.append(row)

lines += [
    "",
    "  SAMPLES PER BIN (a bin with under ~5 samples says almost nothing)",
    "",
    head,
]
for r in range(B - 1, -1, -1):
    row = "  %+.3f " % ((edges[r] + edges[r + 1]) / 2)
    for c in range(B):
        row += "%7d" % grid_n[r, c]
    lines.append(row)

# The verdict that decides whether retraining is worth 45 minutes.
fails = ~ok
pin_fail = float(pin[fails].mean()) if fails.any() else 0.0
pin_ok = float(pin[ok].mean()) if ok.any() else 0.0
lines += [
    "",
    "  arm joints pinned at a limit, mean per env:",
    "     on successes %.2f     on failures %.2f" % (pin_ok, pin_fail),
    "",
]
if pin_fail >= 1.0:
    lines += [
        "  VERDICT: KINEMATIC WALL. Failures sit with joints at their limits, so",
        "  the far cube is outside what this mounting can reach. Retraining cannot",
        "  buy it -- move the robot, shrink the table, or accept the bound.",
    ]
else:
    lines += [
        "  VERDICT: POLICY WALL, not kinematics. Failures happen with joints well",
        "  inside their limits (%.2f pinned on average), so the arm CAN reach where"
        % pin_fail,
        "  it is failing. That is a training-distribution problem, and widening the",
        "  spawn box is the one-variable change that addresses it.",
    ]
lines += [
    "",
    "  NOT a substitute for sweep_shift.sh: this spreads 512 trials over the whole",
    "  box, so a bin holds tens of samples. It says where to aim. The verdict on",
    "  any specific offset still comes from a 512-trial run at that offset.",
]

out = "envelope_%s.txt" % args.label
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

simulation_app.close()
