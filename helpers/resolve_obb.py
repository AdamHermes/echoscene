"""
Drop this file next to eval_3dfront.py, then in eval_3dfront.py:

1. Replace the old resolve_bbox_collisions import/definition with:
       from resolve_obb import resolve_bbox_collisions_obb

2. Replace the call at line ~406:
       boxes_pred_den = resolve_bbox_collisions(boxes_pred_den, objectness_mask=mask)
   with:
       boxes_pred_den = resolve_bbox_collisions_obb(
           boxes_pred_den, angles_pred, objectness_mask=mask
       )
"""

import numpy as np
import torch


# ── OBB helpers ──────────────────────────────────────────────────────────────

def _obb_corners_xz(cx, cz, l, w, angle_deg):
    """4 corners of an OBB footprint in the xz plane."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ax = np.array([ cos_t, sin_t])   # local x-axis → world xz
    az = np.array([-sin_t, cos_t])   # local z-axis → world xz
    c  = np.array([cx, cz])
    hx, hz = l / 2.0, w / 2.0
    return np.stack([
        c + hx*ax + hz*az,
        c + hx*ax - hz*az,
        c - hx*ax + hz*az,
        c - hx*ax - hz*az,
    ])                                # (4, 2)  columns = [x, z]


def _sat_overlap(corners_a, corners_b, axis):
    """Signed overlap of two corner sets projected onto axis (positive = overlapping)."""
    pa = corners_a @ axis
    pb = corners_b @ axis
    return min(pa.max(), pb.max()) - max(pa.min(), pb.min())


def _obb_sat_xz(box_i, ang_i, box_j, ang_j):
    """
    Separating Axis Theorem for two OBBs in the xz floor plane.
    Returns (colliding, min_translation_depth, push_axis_xz).
    push_axis_xz points FROM i TOWARD j (unit vector in xz).
    """
    li, wi = float(box_i[0]), float(box_i[2])
    cxi, czi = float(box_i[3]), float(box_i[5])

    lj, wj = float(box_j[0]), float(box_j[2])
    cxj, czj = float(box_j[3]), float(box_j[5])

    corners_i = _obb_corners_xz(cxi, czi, li, wi, ang_i)
    corners_j = _obb_corners_xz(cxj, czj, lj, wj, ang_j)

    theta_i = np.radians(ang_i)
    theta_j = np.radians(ang_j)

    # 4 candidate separating axes (2 per OBB)
    axes = [
        np.array([ np.cos(theta_i),  np.sin(theta_i)]),  # i local-x
        np.array([-np.sin(theta_i),  np.cos(theta_i)]),  # i local-z
        np.array([ np.cos(theta_j),  np.sin(theta_j)]),  # j local-x
        np.array([-np.sin(theta_j),  np.cos(theta_j)]),  # j local-z
    ]

    min_depth = float('inf')
    push_axis = None

    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        overlap = _sat_overlap(corners_i, corners_j, axis)
        if overlap <= 0:
            return False, 0.0, None          # separating axis found
        if overlap < min_depth:
            min_depth = overlap
            diff = np.array([cxj - cxi, czj - czi])
            sign = 1.0 if np.dot(diff, axis) >= 0 else -1.0
            push_axis = axis * sign

    return True, min_depth, push_axis


# ── public API ────────────────────────────────────────────────────────────────

def resolve_bbox_collisions_obb(
    boxes,                    # (N, 6) tensor  [l, h, w, x, y, z]
    angles_pred,              # (N,) or (N,1) tensor — yaw in DEGREES
    objectness_mask=None,     # (N,) bool tensor — False → skip (floor, _scene_)
    max_iter=80,
    push_eps=0.02,            # extra clearance added after resolving each overlap (m)
    verbose=True,
):
    """
    OBB-aware collision resolver using the Separating Axis Theorem in the xz plane.

    Replaces the old AABB-only resolve_bbox_collisions.  Key differences:
      • Uses the actual yaw angle to build oriented footprints.
      • Push direction comes from the minimum-translation-depth axis (SAT),
        not just the smallest AABB axis — so a rotated wardrobe is pushed
        along its true face normal, not an arbitrary world axis.
      • Only x and z translations are modified; y and sizes are untouched.

    Args:
        boxes:           (N, 6) cuda/cpu tensor [l, h, w, x, y, z].
        angles_pred:     (N,) or (N,1) yaw in degrees.
        objectness_mask: (N,) bool — True for real furniture, False for floor/_scene_.
        max_iter:        maximum SAT-resolve passes.
        push_eps:        small extra gap added after every resolved overlap.
        verbose:         print per-iteration info.

    Returns:
        (N, 6) tensor with updated x/z translations.
    """
    boxes   = boxes.clone()
    N       = boxes.shape[0]
    device  = boxes.device

    if isinstance(angles_pred, torch.Tensor):
        angles_np = angles_pred.detach().cpu().numpy().flatten().astype(float)
    else:
        angles_np = np.array(angles_pred, dtype=float).flatten()

    boxes_np = boxes.detach().cpu().numpy().astype(float)

    for iteration in range(max_iter):
        moved = False

        for i in range(N):
            if objectness_mask is not None and not bool(objectness_mask[i]):
                continue

            for j in range(i + 1, N):
                if objectness_mask is not None and not bool(objectness_mask[j]):
                    continue

                colliding, depth, push_axis = _obb_sat_xz(
                    boxes_np[i], angles_np[i],
                    boxes_np[j], angles_np[j],
                )

                if colliding and push_axis is not None:
                    moved = True
                    shift = (depth + push_eps) / 2.0
                    boxes_np[i, 3] -= shift * push_axis[0]  # x
                    boxes_np[i, 5] -= shift * push_axis[1]  # z
                    boxes_np[j, 3] += shift * push_axis[0]  # x
                    boxes_np[j, 5] += shift * push_axis[1]  # z

        if not moved:
            if verbose:
                print(f"  [OBB resolve] converged at iteration {iteration+1}")
            break
    else:
        if verbose:
            print(f"  [OBB resolve] hit max_iter={max_iter}, residual overlaps may remain")

    # write x and z back to the GPU tensor
    boxes[:, 3] = torch.tensor(boxes_np[:, 3], dtype=boxes.dtype, device=device)
    boxes[:, 5] = torch.tensor(boxes_np[:, 5], dtype=boxes.dtype, device=device)
    return boxes