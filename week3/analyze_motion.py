"""Compare recordings: where the arm goes, relative to the robot base, and where
the cube is. Pure numbers from the recordings — no simulator needed."""
import json, sys, os

D = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:] or ["motion_199680.json", "motion_998400.json", "motion.json"]


def rel(p, base):
    return [p[i] - base[i] for i in range(3)]


def dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


for name in FILES:
    path = os.path.join(D, name)
    if not os.path.exists(path):
        print(f"{name}: MISSING")
        continue
    fr = json.load(open(path))
    if "ee" not in fr[0]:
        print(f"{name}: old format, no ee frame recorded")
        continue

    base = fr[0]["base"]
    cube_b = rel(fr[0]["cube"], base)
    grip = [(f["joints"][7] + f["joints"][8]) / 2 for f in fr]
    shut = next((i for i, g in enumerate(grip) if g < 0.005), None)
    gaps = [dist(f["ee"], f["cube"]) for f in fr]
    imin = gaps.index(min(gaps))

    print(f"=== {name}  ({len(fr)} frames)")
    print(f"  cube, relative to robot base: x {cube_b[0]:+.3f}  y {cube_b[1]:+.3f}  z {cube_b[2]:+.3f}")
    for label, i in [("start", 0), ("fingers shut", shut), ("closest", imin), ("end", len(fr) - 1)]:
        if i is None:
            continue
        ee_b = rel(fr[i]["ee"], base)
        print(f"  {label:13s} f{i:3d}  gripper x {ee_b[0]:+.3f} y {ee_b[1]:+.3f} z {ee_b[2]:+.3f}"
              f"   gap {gaps[i]:.3f} m")
    ee_first, ee_last = rel(fr[0]["ee"], base), rel(fr[-1]["ee"], base)
    print(f"  gripper moved: dx {ee_last[0]-ee_first[0]:+.3f} dy {ee_last[1]-ee_first[1]:+.3f} "
          f"dz {ee_last[2]-ee_first[2]:+.3f} (total {dist(fr[-1]['ee'], fr[0]['ee']):.3f} m)")
    print(f"  cube travelled: {dist(fr[-1]['cube'], fr[0]['cube']):.4f} m")
