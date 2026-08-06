"""Grip geometry over the frames where the cube is genuinely airborne.

The number that separates a real grip from a corner hook is `gap`: the distance
from the cube's centre to the fingertip centre. The cube's half-width is 21 mm,
so a gap comfortably under 21 mm means the cube is BETWEEN the fingers; a gap
above it means the cube is hanging off them and held only by friction.

    stage 16   31.5 mm  -- outside the half-width: a corner hook
    stage 17   12.7 mm  -- a real grip, but the lift collapsed

Reads the CSV written by check_airborne.py. Pure stdlib, no sim needed.

Usage: python grip_geometry.py airborne_s18.csv [more.csv ...]
"""

import csv
import sys

HALF_WIDTH_MM = 21.0
AIRBORNE_M = 0.005


def stats(path):
    rows = list(csv.DictReader(open(path)))
    air = [r for r in rows if float(r["lowest_corner"]) > AIRBORNE_M]
    if not air:
        print("%-22s no airborne frames" % path)
        return
    # A few frames have the cube metres away (dropped, or flung) while its
    # lowest corner is still above 5 mm. Those are not carries and they drag a
    # mean anywhere -- one 340 mm frame moved the average by 1.5 mm. Report the
    # MEDIAN, and say how many frames were outside a plausible carry.
    gap = sorted(float(r["gap"]) * 1000 for r in air)
    tilt = sorted(float(r["tilt_deg"]) for r in air)
    low = sorted(float(r["lowest_corner"]) * 1000 for r in air)
    med = lambda v: v[len(v) // 2]
    loose = sum(1 for g in gap if g > 100.0)
    med_gap = med(gap)
    verdict = "REAL GRIP" if med_gap < HALF_WIDTH_MM else "corner hook"
    print("%-22s frames %3d | gap med %5.1f mm (min %5.1f) | "
          "tilt med %4.1f max %4.1f deg | lift med %5.1f max %5.1f mm | "
          "flung %d | %s"
          % (path, len(air), med_gap, min(gap),
             med(tilt), max(tilt), med(low), max(low), loose, verdict))


paths = sys.argv[1:] or ["airborne_s18.csv"]
print("cube half-width %.0f mm -- gap below that means the cube is BETWEEN the fingers\n"
      % HALF_WIDTH_MM)
for p in paths:
    try:
        stats(p)
    except FileNotFoundError:
        print("%-22s (missing)" % p)
