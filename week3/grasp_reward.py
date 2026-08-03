"""Rewards that make every step toward a grasp pay more than the last.

The stage-9 autopsy found the old design guilty: approach was multiplied by a
hand-matching term, which quietly made "hover 2-3 cm out, hand open" the best
state the robot could actually reach. Every 10M run converged there, and after
the gripper-head reset PPO re-learned "open" from a zeroed head (grip bias
+0.017 -> +0.062 over 3M steps, monotonic) — the gradient itself preferred it.

Three rules replace it:

1. Approach pays with ANY hand state, all the way in. No term is allowed to
   make the robot poorer for standing closer to the cube.
2. Closing pays only where closing means grasping. The pay is a bump that
   peaks where the cube physically stops the fingers (sum 0.042 — measured in
   finger_report.txt), gated smoothly by being at the cube. An empty fist
   scores zero, finger-twitching 4 cm out scores ~zero, and the very first
   close command at the cube already pays (continuous — no cliffs, the
   lesson replays taught us twice).
3. Height still pays only while the cube is truly held — and "held" now means
   the fingers are stopped BY the cube, which an empty shut fist (sum ~0.004)
   can no longer satisfy. The old gate (sum < 0.06) counted a fist clenched
   next to the cube as a hold.

The staircase this buys, in reward per step: hover at 3 cm ~3.9 -> open hand
at 5 mm ~5.5 -> grasped ~11 -> lifted and carrying ~31. Monotonic. The valley
that swallowed stages 1-9 is gone.
"""

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

# fingers are commanded open (0.04 each) or shut (0.0); the 4.2 cm cube stops
# them at a joint sum of ~0.042, an empty fist reaches ~0.004
OPEN_SUM = 0.08
GRASP_SUM = 0.042


def _finger_sum(robot: Articulation) -> torch.Tensor:
    return robot.data.joint_pos[:, -2:].sum(dim=1)


def _on_cube_bump(fsum: torch.Tensor) -> torch.Tensor:
    """1.0 where the cube stops the fingers, falling to 0 at fully open AND at
    an empty fist. Rewarding the stop point instead of raw closure is what
    makes a clenched empty fist worthless."""
    return (1.0 - (fsum - GRASP_SUM).abs() / (OPEN_SUM - GRASP_SUM)).clamp(min=0.0)


def _ee_cube_distance(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(obj.data.root_pos_w - ee_pos, dim=1)


def _held(
    env: ManagerBasedRLEnv,
    hold_distance: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """True where the cube is at the gripper AND the fingers are stopped by it.
    bump > 0.5 means the finger sum is within ~2 mm of the cube-stop point —
    an empty fist (~0.004) and an open hand (0.08) both fail it."""
    robot: Articulation = env.scene[robot_cfg.name]
    near = _ee_cube_distance(env, object_cfg, ee_frame_cfg) < hold_distance
    stopped_by_cube = _on_cube_bump(_finger_sum(robot)) > 0.5
    return near & stopped_by_cube


def approach_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Pull the gripper to the cube, hand state ignored. Coarse ramp carries it
    across the table; the fine ramp keeps the slope alive over the last few
    centimetres, so at 3 cm the robot still earns ~+1.5/step by coming in."""
    distance = _ee_cube_distance(env, object_cfg, ee_frame_cfg)
    coarse = 1.0 - torch.tanh(distance / 0.25)
    fine = 1.0 - torch.tanh(distance / 0.03)
    return 0.6 * coarse + 0.4 * fine


def fingers_on_cube(
    env: ManagerBasedRLEnv,
    in_position_std: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay for closing exactly where closing means grasping.

    The position gate (1 - tanh(d / 0.015)) is ~1 with the TCP on the cube and
    ~0.04 at 3 cm, so the twitch-at-a-distance exploit of stage 9 earns
    nothing. The bump term rises along the whole close trajectory at the cube
    (first close command: 0.22, second: 0.44, cube-stop: 1.0), so a single
    exploratory close at the right place is immediately reinforced."""
    robot: Articulation = env.scene[robot_cfg.name]
    distance = _ee_cube_distance(env, object_cfg, ee_frame_cfg)
    in_position = 1.0 - torch.tanh(distance / in_position_std)
    return in_position * _on_cube_bump(_finger_sum(robot))


def object_lifted_in_hand(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_distance: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The stock height check, but only while the cube is held."""
    lifted = lift_mdp.object_is_lifted(env, minimal_height=minimal_height, object_cfg=object_cfg)
    return lifted * _held(env, hold_distance, object_cfg, ee_frame_cfg, robot_cfg).float()


def object_goal_distance_in_hand(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    hold_distance: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Carrying the cube toward the goal, counted only while it is held."""
    tracking = lift_mdp.object_goal_distance(
        env, std=std, minimal_height=minimal_height, command_name=command_name,
        robot_cfg=robot_cfg, object_cfg=object_cfg,
    )
    return tracking * _held(env, hold_distance, object_cfg, ee_frame_cfg, robot_cfg).float()
