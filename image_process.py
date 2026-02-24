import glob
import math
import os
import re

import cv2
import numpy as np
from skimage.morphology import skeletonize

# =========================
# CONFIG
# =========================
STROKE_DIR = "strokes"
PX_PER_MM = 23.6

DRAW_Z = 0.0
LIFT_Z = 20.0
MIN_STEP_MM = 1.0

RAPID_FILE = "DrawImage.mod"

# =========================
# RAPID CONSTANTS
# =========================
ORIENT = "[0,-0.7071068,-0.7071068,0]"
CONF = "[0,0,-1,0]"
EXTAX = "[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]"


# =========================
# UTILS
# =========================
def robtarget(x, y, z):
    return f"[[{x:.3f},{y:.3f},{z:.3f}],{ORIENT},{CONF},{EXTAX}]"


def dist_px(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_mm(a, b):
    return dist_px(a, b) / PX_PER_MM


# =========================
# SKELETON HELPERS
# =========================
def get_neighbors(x, y, skel):
    n = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                if skel[ny, nx]:
                    n.append((nx, ny))
    return n


def trace_skeleton(skel):
    ys, xs = np.where(skel)
    pixels = set(zip(xs, ys))

    # endpoints
    endpoints = [p for p in pixels if len(get_neighbors(*p, skel)) == 1]

    if endpoints:
        start = endpoints[0]
    else:
        start = next(iter(pixels))  # loop stroke

    stroke = []
    visited = set()
    curr = start
    prev = None

    while True:
        stroke.append(curr)
        visited.add(curr)

        nbrs = get_neighbors(*curr, skel)
        nxt = [p for p in nbrs if p != prev and p not in visited]

        if not nxt:
            break

        prev = curr
        curr = nxt[0]

    return stroke


# =========================
# LOAD + PROCESS STROKES
# =========================
def layer_index(path):
    # extract the number from "Paint Layer x.PNG"
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


stroke_files = sorted(glob.glob(os.path.join(STROKE_DIR, "*.PNG")), key=layer_index)
if not stroke_files:
    raise RuntimeError("No stroke PNGs found in strokes/")

print(f"Found {len(stroke_files)} stroke layers")

strokes = []

for fn in stroke_files:
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    # --- extract mask ---
    if img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray < 250

    if np.count_nonzero(mask) < 10:
        continue

    # --- skeletonize ---
    skel = skeletonize(mask)

    # --- trace centerline ---
    stroke = trace_skeleton(skel)

    if len(stroke) > 2:
        strokes.append(stroke)

print(f"Valid strokes after skeletonizing: {len(strokes)}")


canvas_h = mask.shape[0]
canvas_w = mask.shape[1]

stroke_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

rng = np.random.default_rng(42)

for stroke in strokes:
    color = rng.integers(80, 255, size=3).tolist()

    for i in range(len(stroke) - 1):
        x1, y1 = stroke[i]
        x2, y2 = stroke[i + 1]

        cv2.line(
            stroke_img,
            (x1, y1),
            (x2, y2),
            color,
            thickness=1,  # centerline
            lineType=cv2.LINE_AA,
        )

cv2.imwrite("strokes.png", stroke_img)
print("Stroke centerlines saved: strokes.png")


def choose_stroke_direction(stroke, current_pos):
    """
    Decide whether to reverse stroke to minimize travel distance
    current_pos: (x_px, y_px) or None
    """
    if current_pos is None:
        return stroke

    d_start = dist_px(current_pos, stroke[0])
    d_end = dist_px(current_pos, stroke[-1])

    if d_end < d_start:
        return list(reversed(stroke))
    return stroke


# =========================
# WRITE RAPID MODULE
# =========================
def img_to_cartesian(pt, H):
    x, y = pt
    return (x, H - 1 - y)


with open(RAPID_FILE, "w") as f:
    f.write("MODULE DrawImage\n\n")

    f.write("  PROC draw_home(PERS wobjdata wobjDraw, PERS tooldata toolPen)\n")
    f.write(
        "    MoveAbsJ [[0,45,0,0,45,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]], v100, fine, toolPen;\n"
    )
    f.write(
        f"    MoveL {robtarget(0,0,LIFT_Z)}, v100, fine, toolPen\\WObj:=wobjDraw;\n"
    )
    f.write("  ENDPROC\n\n")

    f.write("  PROC draw_image(PERS wobjdata wobjDraw, PERS tooldata toolPen)\n\n")

    current_px_pos = None

    for stroke in strokes:
        stroke = choose_stroke_direction(stroke, current_px_pos)

        x0_px, y0_px = img_to_cartesian(stroke[0], canvas_h)
        x0 = x0_px / PX_PER_MM
        y0 = y0_px / PX_PER_MM

        # Move above
        f.write(
            f"    MoveL {robtarget(x0, y0, LIFT_Z)}, v100, fine, toolPen\\WObj:=wobjDraw;\n"
        )

        # Pen down
        f.write(
            f"    MoveL {robtarget(x0, y0, DRAW_Z)}, v200, fine, toolPen\\WObj:=wobjDraw;\n"
        )
        f.write(f"    WaitTime 0.2;\n")

        last_px = stroke[0]

        for x_px, y_px in stroke[1:]:
            x_px, y_px = img_to_cartesian((x_px, y_px), canvas_h)

            if dist_mm((x_px, y_px), last_px) < MIN_STEP_MM:
                continue

            x = x_px / PX_PER_MM
            y = y_px / PX_PER_MM

            f.write(
                f"    MoveL {robtarget(x, y, DRAW_Z)}, v100, fine, toolPen\\WObj:=wobjDraw;\n"
            )
            last_px = (x_px, y_px)

        # Draw last point
        x_px, y_px = img_to_cartesian(stroke[-1], canvas_h)
        x = x_px / PX_PER_MM
        y = y_px / PX_PER_MM

        f.write(
            f"    MoveL {robtarget(x, y, DRAW_Z)}, v100, fine, toolPen\\WObj:=wobjDraw;\n"
        )

        # Pen up
        x_px, y_px = img_to_cartesian(stroke[-1], canvas_h)
        x = x_px / PX_PER_MM
        y = y_px / PX_PER_MM

        f.write(f"    WaitTime 0.1;\n")
        f.write(
            f"    MoveL {robtarget(x, y, LIFT_Z)}, v200, fine, toolPen\\WObj:=wobjDraw;\n\n"
        )

        current_px_pos = stroke[-1]

    f.write("  ENDPROC\n")
    f.write("ENDMODULE\n")

print("✅ RAPID module generated:", RAPID_FILE)
