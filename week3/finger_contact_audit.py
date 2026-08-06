"""Are the fingers actually TOUCHING the cube while it is 'held'?

The screenshot shows a finger visually inside the cube, which should be
impossible. This checks the numbers behind that picture.

The arithmetic that matters:

    cube edge                     0.042 m
    finger_sum = j1 + j2, and each finger joint is its distance from the
    gripper centreline, so the OPENING between the finger inner faces is
    exactly finger_sum.

    finger_sum == 0.042  ->  fingers exactly on the cube faces
    finger_sum <  0.042  ->  fingers would be INSIDE the cube (penetration)
    finger_sum >  0.042  ->  clearance: NOT TOUCHING, so no friction, so
                             nothing to hold the cube up

`_on_cube_bump` in grasp_reward.py accepts |finger_sum - 0.042| < 0.012, i.e.
anything from 0.030 to 0.054. That band is +/- 12 mm on a 42 mm cube. At the
top of it the fingers are 6 mm clear of each face and touching nothing at all,
yet it still counts as GRASPED. That is the thing to test.

Reads the CSVs written by check_airborne.py. Pure stdlib.
"""

import csv
import sys

CUBE = 0.042
BAND = 0.012        # the tolerance _on_cube_bump actually uses
AIRBORNE = 0.005


def audit(path):
    rows = list(csv.DictReader(open(path)))
    air = [r for r in rows if float(r["lowest_corner"]) > AIRBORNE]
    if not air:
        print("%-20s no airborne frames" % path)
        return

    fs = sorted(float(r["finger_sum"]) for r in air)
    med = fs[len(fs) // 2]
    # Clearance per face, in mm: how far each finger sits OFF the cube surface.
    clear_mm = (med - CUBE) / 2 * 1000
    touching = sum(1 for f in fs if f <= CUBE + 0.001)
    penetrating = sum(1 for f in fs if f < CUBE - 0.001)

    if med < CUBE - 0.001:
        verdict = "PENETRATING the cube"
    elif med <= CUBE + 0.001:
        verdict = "in contact"
    else:
        verdict = "NOT TOUCHING (%.1f mm clear per face)" % clear_mm

    print("%-20s airborne %3d | finger_sum med %.4f (min %.4f max %.4f) | "
          "frames in contact %3d/%3d | penetrating %3d | %s"
          % (path, len(air), med, fs[0], fs[-1], touching, len(fs),
             penetrating, verdict))


print("cube edge %.3f m. finger_sum IS the opening between the fingers." % CUBE)
print("_on_cube_bump accepts %.3f to %.3f -- at the top of that band the"
      % (CUBE - BAND, CUBE + BAND))
print("fingers are %.0f mm clear of each face and touching nothing.\n"
      % (BAND / 2 * 1000))

for p in sys.argv[1:] or ["airborne_s16.csv"]:
    try:
        audit(p)
    except FileNotFoundError:
        print("%-20s (missing)" % p)
