"""
Visualize GT bounding boxes for a room from the 3D-FRONT dataset.

Usage:
    python visualize_gt_boxes.py --room Bedroom-6482
    python visualize_gt_boxes.py --room Bedroom-6482 --split trainval --view 3d
    python visualize_gt_boxes.py --room Bedroom-6482 --room_type bedroom --list_rooms
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FRONT_DIR = os.path.join(os.path.dirname(__file__), "FRONT")

CATEGORY_COLORS = {
    "bed":           "#3498db",
    "double_bed":    "#2980b9",
    "single_bed":    "#5dade2",
    "wardrobe":      "#2ecc71",
    "nightstand":    "#9b59b6",
    "chair":         "#e74c3c",
    "table":         "#e67e22",
    "desk":          "#f39c12",
    "sofa":          "#1abc9c",
    "ceiling_lamp":  "#f1c40f",
    "lamp":          "#f1c40f",
    "floor":         "#ecf0f1",
    "_scene_":       "#bdc3c7",
}

ROOM_TYPE_MAP = {
    "bedroom":    "bedroom",
    "diningroom": "diningroom",
    "livingroom": "livingroom",
    "all":        "all",
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the Y axis (3D-FRONT convention)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, -s],
                     [0, 1,  0],
                     [s, 0,  c]])


def param7_to_corners_3d(param7) -> np.ndarray:
    """
    Convert param7 = [l, h, w, cx, cy, cz, angle_rad] to 8 corners in 3D.

    Coordinate system (3D-FRONT):
        X = right, Y = up, Z = depth
        angle = rotation around Y axis

    Returns shape (8, 3).
    """
    l, h, w, cx, cy, cz, angle = param7
    # Local box corners before rotation: (l along X, h along Y, w along Z)
    half_l, half_h, half_w = l / 2, h, w / 2  # h is full height, base at y=0
    local_corners = np.array([
        [-half_l, 0,       -half_w],
        [ half_l, 0,       -half_w],
        [ half_l, 0,        half_w],
        [-half_l, 0,        half_w],
        [-half_l, half_h * 2, -half_w],  # top face
        [ half_l, half_h * 2, -half_w],
        [ half_l, half_h * 2,  half_w],
        [-half_l, half_h * 2,  half_w],
    ])
    R = get_rotation_y(angle)
    rotated = (R @ local_corners.T).T
    center = np.array([cx, cy, cz])
    return rotated + center


def param7_to_footprint_2d(param7):
    """Return the 4 XZ footprint corners (top-down view) of a param7 bbox."""
    l, h, w, cx, cy, cz, angle = param7
    half_l, half_w = l / 2, w / 2
    local = np.array([
        [-half_l, -half_w],
        [ half_l, -half_w],
        [ half_l,  half_w],
        [-half_l,  half_w],
    ])
    c, s = np.cos(angle), np.sin(angle)
    R2d = np.array([[c, -s], [s, c]])
    rotated = (R2d @ local.T).T
    return rotated + np.array([cx, cz])


def obb_edges_3d():
    """12 edges of a box (by corner index pairs). Corners ordered bottom then top."""
    bottom = [0, 1, 2, 3]
    top    = [4, 5, 6, 7]
    edges  = []
    # bottom face ring
    for i in range(4):
        edges.append((bottom[i], bottom[(i+1) % 4]))
    # top face ring
    for i in range(4):
        edges.append((top[i], top[(i+1) % 4]))
    # vertical pillars
    for i in range(4):
        edges.append((bottom[i], top[i]))
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def guess_room_type(room_id: str) -> str:
    rid = room_id.lower()
    for rt in ("bedroom", "diningroom", "livingroom"):
        if rt.replace("room", "").replace("dining", "diningroom") in rid or rt in rid:
            return rt
    return "all"


def load_scene(room_id: str, front_dir: str, split: str, room_type: str):
    """
    Load param7 bboxes + object labels for a given room_id.
    Returns:
        objects: list of dicts {name, param7, color}
        scene_center: np.array [3]
    """
    # Try to figure out room_type from the ID if not supplied
    if room_type is None:
        room_type = guess_room_type(room_id)

    box_file = os.path.join(front_dir, f"obj_boxes_{room_type}_{split}.json")
    rel_file = os.path.join(front_dir, f"relationships_{room_type}_{split}.json")

    if not os.path.exists(box_file):
        sys.exit(f"[ERROR] Box file not found: {box_file}\n"
                 f"  Try --room_type all  or  --split test")
    if not os.path.exists(rel_file):
        sys.exit(f"[ERROR] Relationship file not found: {rel_file}")

    with open(box_file, "r") as f:
        box_data = json.load(f)
    with open(rel_file, "r") as f:
        rel_data = json.load(f)

    if room_id not in box_data:
        available = [k for k in box_data if room_id.lower() in k.lower()]
        msg = f"[ERROR] Room '{room_id}' not found in {box_file}."
        if available:
            msg += f"\n  Did you mean one of: {available[:10]}"
        sys.exit(msg)

    scene_box = box_data[room_id]
    scene_center = np.array(scene_box["scene_center"])

    # Build id → label map from relationships json
    label_map = {}
    for scan in rel_data["scans"]:
        if scan["scan"] == room_id:
            for k, v in scan["objects"].items():
                label_map[int(k)] = v
            break

    objects = []
    for key, info in scene_box.items():
        if key == "scene_center":
            continue
        obj_id = int(key)
        name = label_map.get(obj_id, f"obj_{obj_id}")
        param7 = np.array(info["param7"], dtype=float)
        color = CATEGORY_COLORS.get(name, "#aaaaaa")
        objects.append({"id": obj_id, "name": name, "param7": param7, "color": color})

    return objects, scene_center


def list_rooms(front_dir: str, room_type: str, split: str):
    box_file = os.path.join(front_dir, f"obj_boxes_{room_type}_{split}.json")
    with open(box_file) as f:
        data = json.load(f)
    rooms = [k for k in data if k != "scene_center"]
    print(f"\nRooms in {box_file} ({len(rooms)} total):\n")
    for r in sorted(rooms):
        print(f"  {r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2-D top-down visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_2d(objects, scene_center, room_id: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    legend_handles = []
    seen_labels = set()

    for obj in objects:
        p7 = obj["param7"].copy()
        p7[3:6] -= scene_center          # center-normalize (same as dataset)
        name   = obj["name"]
        color  = obj["color"]
        is_floor = (name in ("floor", "_scene_"))

        footprint = param7_to_footprint_2d(p7)  # (4, 2) in XZ

        alpha   = 0.15 if is_floor else 0.65
        lw      = 1.5  if is_floor else 2.0
        ec      = "#ffffff" if not is_floor else "#555577"

        poly = mpatches.Polygon(
            footprint, closed=True,
            facecolor=color, edgecolor=ec,
            alpha=alpha, linewidth=lw, zorder=2 if not is_floor else 1
        )
        ax.add_patch(poly)

        # Forward direction arrow (along -Z local = object front)
        cx, cz = p7[3], p7[5]
        angle  = p7[6]
        arrow_len = min(p7[0], p7[2]) * 0.35
        dx = -np.sin(angle) * arrow_len
        dz =  np.cos(angle) * arrow_len
        if not is_floor:
            ax.annotate(
                "", xy=(cx + dx, cz + dz), xytext=(cx, cz),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
                zorder=4
            )

        # Label
        if not is_floor:
            short = name.replace("_", " ")
            ax.text(
                cx, cz, short,
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
                zorder=5
            )

        # Legend
        display_name = name.replace("_", " ").title()
        if display_name not in seen_labels:
            seen_labels.add(display_name)
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="white", alpha=0.8,
                               linewidth=0.8, label=display_name)
            )

    # Scene origin marker
    ax.scatter(0, 0, c="red", s=80, zorder=6, label="scene center")
    ax.text(0.05, 0.05, "⊕", fontsize=10, color="red",
            transform=ax.transAxes, va="bottom")

    # Axis styling
    ax.set_aspect("equal")
    ax.grid(True, color="#334455", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")

    ax.set_xlabel("X  (right) →", color="#aabbcc", fontsize=10)
    ax.set_ylabel("Z  (depth) →", color="#aabbcc", fontsize=10)
    ax.set_title(
        f"GT Bounding Boxes — {room_id}  (top-down, scene-centered)",
        color="white", fontsize=13, fontweight="bold", pad=14
    )

    legend = ax.legend(
        handles=legend_handles, loc="upper right",
        framealpha=0.35, facecolor="#1a1a2e",
        edgecolor="#445566", labelcolor="white", fontsize=8
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3-D visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_3d(objects, scene_center, room_id: str):
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")

    edges = obb_edges_3d()
    legend_handles = []
    seen_labels = set()

    all_pts = []

    for obj in objects:
        p7 = obj["param7"].copy()
        p7[3:6] -= scene_center
        name  = obj["name"]
        color = obj["color"]
        is_floor = (name in ("floor", "_scene_"))

        corners = param7_to_corners_3d(p7)   # (8, 3)
        all_pts.append(corners)

        # Draw 12 edges
        edge_color = "#888899" if is_floor else "white"
        edge_alpha = 0.25 if is_floor else 0.85
        for e0, e1 in edges:
            xs = [corners[e0, 0], corners[e1, 0]]
            ys = [corners[e0, 1], corners[e1, 1]]
            zs = [corners[e0, 2], corners[e1, 2]]
            ax.plot(xs, ys, zs, color=edge_color, alpha=edge_alpha, linewidth=1.2)

        # Fill faces (top & bottom) with semi-transparent color
        if not is_floor:
            # Bottom face (indices 0-3), Top face (indices 4-7)
            for face_idx in [[0, 1, 2, 3], [4, 5, 6, 7]]:
                verts = [corners[face_idx]]
                poly  = Poly3DCollection(verts, alpha=0.25)
                poly.set_facecolor(color)
                poly.set_edgecolor("none")
                ax.add_collection3d(poly)
            # Side faces
            for i in range(4):
                face = [corners[[i, (i+1)%4, (i+1)%4+4, i+4]]]
                poly  = Poly3DCollection(face, alpha=0.15)
                poly.set_facecolor(color)
                poly.set_edgecolor("none")
                ax.add_collection3d(poly)

        # Center dot + label
        cx, cy, cz = p7[3], p7[4], p7[5]
        ax.scatter([cx], [cy], [cz], color=color, s=20, zorder=3, depthshade=False)
        if not is_floor:
            short = name.replace("_", " ")
            ax.text(cx, cy + p7[1] * 1.05, cz, short,
                    fontsize=7, color="white", ha="center",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

        display_name = name.replace("_", " ").title()
        if display_name not in seen_labels:
            seen_labels.add(display_name)
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="white",
                               alpha=0.8, linewidth=0.8, label=display_name)
            )

    # Auto-scale axes
    if all_pts:
        all_pts = np.vstack(all_pts)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        padding = 0.3
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)

    ax.set_xlabel("X →", color="#aabbcc", fontsize=9)
    ax.set_ylabel("Y (up) →", color="#aabbcc", fontsize=9)
    ax.set_zlabel("Z →", color="#aabbcc", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#334455")
    ax.yaxis.pane.set_edgecolor("#334455")
    ax.zaxis.pane.set_edgecolor("#334455")
    ax.grid(True, color="#334455", linewidth=0.4, alpha=0.5)

    ax.set_title(
        f"GT Bounding Boxes — {room_id}  (3D, scene-centered)",
        color="white", fontsize=12, fontweight="bold", pad=14
    )
    ax.legend(
        handles=legend_handles, loc="upper left",
        framealpha=0.35, facecolor="#1a1a2e",
        edgecolor="#445566", labelcolor="white", fontsize=8
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT bounding boxes for a 3D-FRONT room."
    )
    parser.add_argument(
        "--room", type=str, default="Bedroom-6482",
        help="Room ID, e.g. 'Bedroom-6482' (default: Bedroom-6482)"
    )
    parser.add_argument(
        "--front_dir", type=str, default=DEFAULT_FRONT_DIR,
        help=f"Path to FRONT/ data dir (default: {DEFAULT_FRONT_DIR})"
    )
    parser.add_argument(
        "--room_type", type=str, default=None,
        choices=["bedroom", "diningroom", "livingroom", "all"],
        help="Room type for choosing the right JSON files. Auto-detected if not set."
    )
    parser.add_argument(
        "--split", type=str, default="trainval",
        choices=["trainval", "test"],
        help="Dataset split (default: trainval)"
    )
    parser.add_argument(
        "--view", type=str, default="both",
        choices=["2d", "3d", "both"],
        help="Visualization mode: top-down 2D, 3D OBBs, or both (default: both)"
    )
    parser.add_argument(
        "--list_rooms", action="store_true",
        help="List all room IDs available in the chosen split/room_type and exit."
    )
    args = parser.parse_args()

    room_type = args.room_type or guess_room_type(args.room)

    if args.list_rooms:
        list_rooms(args.front_dir, room_type, args.split)
        return

    print(f"\n📦  Loading: {args.room}  (room_type={room_type}, split={args.split})")
    objects, scene_center = load_scene(args.room, args.front_dir, args.split, room_type)

    print(f"\n{'─'*55}")
    print(f"{'ID':>4}  {'Name':<20}  {'l':>6}  {'h':>6}  {'w':>6}  {'cx':>7}  {'cz':>7}  {'angle°':>8}")
    print(f"{'─'*55}")
    for obj in objects:
        p = obj["param7"]
        cx_rel = p[3] - scene_center[0]
        cz_rel = p[5] - scene_center[2]
        angle_deg = np.degrees(p[6])
        print(f"  {obj['id']:>2}  {obj['name']:<20}  {p[0]:>6.3f}  {p[1]:>6.3f}  "
              f"{p[2]:>6.3f}  {cx_rel:>7.3f}  {cz_rel:>7.3f}  {angle_deg:>8.2f}°")
    print(f"{'─'*55}")
    print(f"  Scene center: {scene_center}\n")

    if args.view in ("2d", "both"):
        visualize_2d(objects, scene_center, args.room)
    if args.view in ("3d", "both"):
        visualize_3d(objects, scene_center, args.room)


if __name__ == "__main__":
    main()
