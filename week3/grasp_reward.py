"""Custom reward: pays the robot when the gripper is BOTH close to the cube
AND closed around it. This is the missing staircase step between reach and lift."""

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv


def object_is_grasped(
    env: ManagerBasedRLEnv,
    grasp_distance: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward = 1.0 when gripper is near the cube AND fingers are closing."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # 1. Is the gripper close to the cube?
    cube_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos - ee_pos, dim=1)
    is_close = distance < grasp_distance

    # 2. Are the fingers closing? (Franka gripper joints are the last 2)
    finger_pos = robot.data.joint_pos[:, -2:]          # both finger joints
    finger_opening = finger_pos.sum(dim=1)             # small = closed
    is_closing = finger_opening < 0.06                 # closed threshold

    # Reward only when BOTH are true
    return (is_close & is_closing).float()
