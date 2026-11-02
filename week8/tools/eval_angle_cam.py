# Run lerobot-eval, but record videos from the "angle" camera with a nicer look.
# The policy's observation still comes from the "top" camera with the original colors:
# colors are changed only inside render() and restored before returning.
import numpy as np
import gym_aloha.env as aloha_env
from lerobot.scripts.lerobot_eval import main

H, W = 480, 640
GRAPHITE = [0.16, 0.17, 0.19, 1]
SILVER = [0.62, 0.64, 0.67, 1]
ACCENT = {"vx300s_left": [0.10, 0.72, 0.62, 1], "vx300s_right": [0.20, 0.48, 0.95, 1]}
TABLE = [0.55, 0.53, 0.50, 1]

# Vertical gradient for the empty background (dark slate at top, lighter at bottom).
_top, _bot = np.array([18, 22, 30]), np.array([58, 66, 80])
BG = (_top + (_bot - _top) * np.linspace(0, 1, H)[:, None]).astype(np.uint8)[:, None, :].repeat(W, 1)


def style(model):
    rgba = model.geom_rgba.copy()
    for g in range(model.ngeom):
        name = model.geom(g).name
        arm = name.split("/")[0]
        if name == "table":
            rgba[g] = TABLE
        elif arm in ACCENT:
            part = name.split("/")[1]
            if "gripper" in part or "finger" in part:
                rgba[g] = ACCENT[arm]
            elif part.startswith(("3_", "5_")):
                rgba[g] = SILVER
            else:
                rgba[g] = GRAPHITE
    return rgba


def render(self):
    physics = self._env.physics
    original = physics.model.geom_rgba.copy()
    physics.model.geom_rgba[:] = style(physics.model)
    try:
        rgb = physics.render(height=H, width=W, camera_id="angle")
        depth = physics.render(height=H, width=W, camera_id="angle", depth=True)
    finally:
        physics.model.geom_rgba[:] = original
    background = depth > 5.0
    rgb = rgb.copy()
    rgb[background] = BG[background]
    return rgb


aloha_env.AlohaEnv.render = render

if __name__ == "__main__":
    main()
