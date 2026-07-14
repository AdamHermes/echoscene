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

def _obb_sat_xz(box_i, ang_i, box_j, ang_j, alignment_bias=2.0):
    """
    Separating Axis Theorem for two OBBs in the xz floor plane.
    Returns (colliding, translation_depth, push_axis_xz).
    push_axis_xz points FROM i TOWARD j (unit vector in xz).

    Instead of always choosing the axis with minimum overlap (MTV), we
    prefer the axis that is best aligned with the center-to-center
    direction so that objects are pushed *apart* rather than sideways
    or behind each other.

    Each candidate axis is scored as:
        score = depth / (alignment ** alignment_bias + eps)
    where alignment = |dot(signed_axis, c2c_unit)|.  Lower score wins.
    This favours axes that are both shallow *and* point away from each
    other.  ``alignment_bias`` controls how strongly alignment is
    weighted (higher = stronger preference for center-aligned axes).
    When no SAT axis is reasonably aligned (all < 0.15), we fall back
    to pushing along the raw center-to-center direction using the
    minimum overlap depth.
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

    # Center-to-center direction (from i toward j)
    c2c = np.array([cxj - cxi, czj - czi])
    c2c_len = np.linalg.norm(c2c)
    if c2c_len > 1e-8:
        c2c_unit = c2c / c2c_len
    else:
        c2c_unit = None                      # nearly coincident → fallback later

    # First pass: check for separation & collect per-axis info
    axis_info = []                           # (overlap, signed_axis)
    min_depth = float('inf')

    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        overlap = _sat_overlap(corners_i, corners_j, axis)
        if overlap <= 0:
            return False, 0.0, None          # separating axis found
        # orient axis so it points from i toward j
        if c2c_unit is not None:
            sign = 1.0 if np.dot(c2c, axis) >= 0 else -1.0
        else:
            sign = 1.0
        signed_axis = axis * sign
        axis_info.append((overlap, signed_axis))
        if overlap < min_depth:
            min_depth = overlap

    # ── Axis selection ──────────────────────────────────────────────
    if c2c_unit is None:
        # Objects are nearly coincident: just use the shallowest axis
        best_axis = min(axis_info, key=lambda x: x[0])
        return True, best_axis[0], best_axis[1]

    ALIGN_THRESHOLD = 0.15
    best_score  = float('inf')
    best_depth  = min_depth
    best_push   = None

    for overlap, signed_axis in axis_info:
        alignment = abs(np.dot(signed_axis, c2c_unit))
        if alignment < ALIGN_THRESHOLD:
            continue                          # nearly perpendicular → skip
        score = overlap / (alignment ** alignment_bias + 1e-12)
        if score < best_score:
            best_score = score
            best_depth = overlap
            best_push  = signed_axis

    if best_push is not None:
        return True, best_depth, best_push

    # Fallback: no SAT axis is well-aligned → push along center-to-center
    # using the minimum overlap depth (safe, always separates)
    return True, min_depth, c2c_unit

# ── public API ────────────────────────────────────────────────────────────────

def _has_any_collision(idx, boxes_np, angles_np, objectness_mask, N):
    """Check if object ``idx`` collides with any other active object."""
    for other in range(N):
        if other == idx:
            continue
        if objectness_mask is not None and not bool(objectness_mask[other]):
            continue
        colliding, _, _ = _obb_sat_xz(
            boxes_np[idx], angles_np[idx],
            boxes_np[other], angles_np[other],
        )
        if colliding:
            return True
    return False


def resolve_bbox_collisions_obb(
    boxes,                    # (N, 6) tensor  [l, h, w, x, y, z]
    angles_pred,              # (N,) or (N,1) tensor — yaw in DEGREES
    objectness_mask=None,     # (N,) bool tensor — False → skip (floor, _scene_)
    max_iter=80,
    push_eps=0.02,            # extra clearance added after resolving each overlap (m)
    snap_back_steps=6,        # binary search precision for snap-back (2^6 = 64 levels)
    verbose=True,
):
    """
    Resolve collisions between oriented bounding boxes in two phases:

    **Phase 1 — Collision resolution** (iterative, volume-weighted push):
    Push colliding pairs apart until no overlaps remain.  Larger objects
    (by floor footprint l*w) act as heavier anchors and absorb less
    displacement, so a small desk moves while a large bed stays put.

    **Phase 2 — Snap-back** (post-convergence, collision-safe):
    After all collisions are resolved, each object is moved back toward
    its *original* diffusion-predicted position as far as possible without
    re-introducing any collision.  A binary search finds the maximum safe
    fraction of the return trip.  This minimises displacement from the
    diffusion-predicted layout while guaranteeing collision-freedom.
    """
    boxes   = boxes.clone()
    N       = boxes.shape[0]
    device  = boxes.device

    if isinstance(angles_pred, torch.Tensor):
        angles_np = angles_pred.detach().cpu().numpy().flatten().astype(float)
    else:
        angles_np = np.array(angles_pred, dtype=float).flatten()

    boxes_np = boxes.detach().cpu().numpy().astype(float)

    # ── remember original positions as anchors ─────────────────────
    anchor_x = boxes_np[:, 3].copy()
    anchor_z = boxes_np[:, 5].copy()

    # ── precompute floor-plane areas (l × w) for mass weighting ────
    volumes = np.maximum(boxes_np[:, 0] * boxes_np[:, 2], 1e-6)   # l * w

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: resolve all collisions (no anchor spring)
    # ══════════════════════════════════════════════════════════════════
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
                    total_shift = depth + push_eps

                    # Volume-weighted: heavier object moves less
                    vi, vj = volumes[i], volumes[j]
                    ratio_i = vj / (vi + vj)      # small obj → large ratio
                    ratio_j = vi / (vi + vj)

                    boxes_np[i, 3] -= total_shift * ratio_i * push_axis[0]
                    boxes_np[i, 5] -= total_shift * ratio_i * push_axis[1]
                    boxes_np[j, 3] += total_shift * ratio_j * push_axis[0]
                    boxes_np[j, 5] += total_shift * ratio_j * push_axis[1]

        if not moved:
            if verbose:
                print(f"  [OBB resolve] converged at iteration {iteration+1}")
            break
    else:
        if verbose:
            print(f"  [OBB resolve] hit max_iter={max_iter}, residual overlaps may remain")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: snap-back toward original positions (collision-safe)
    # ══════════════════════════════════════════════════════════════════
    #  For each object, binary-search the largest fraction α ∈ [0, 1]
    #  such that moving it by α · (anchor − current) does not collide
    #  with any other object.  This minimises displacement from the
    #  diffusion-predicted layout while guaranteeing collision-freedom.
    # ──────────────────────────────────────────────────────────────────
    snap_count = 0
    for k in range(N):
        if objectness_mask is not None and not bool(objectness_mask[k]):
            continue

        drift_x = anchor_x[k] - boxes_np[k, 3]
        drift_z = anchor_z[k] - boxes_np[k, 5]

        # skip if the object barely moved
        if abs(drift_x) < 1e-4 and abs(drift_z) < 1e-4:
            continue

        # binary search for the largest safe alpha
        lo, hi = 0.0, 1.0
        saved_x = boxes_np[k, 3]
        saved_z = boxes_np[k, 5]

        for _ in range(snap_back_steps):
            mid = (lo + hi) / 2.0
            boxes_np[k, 3] = saved_x + mid * drift_x
            boxes_np[k, 5] = saved_z + mid * drift_z

            if _has_any_collision(k, boxes_np, angles_np, objectness_mask, N):
                hi = mid          # too far → pull back
            else:
                lo = mid          # safe → try going further

        # commit the largest safe position
        boxes_np[k, 3] = saved_x + lo * drift_x
        boxes_np[k, 5] = saved_z + lo * drift_z
        if lo > 0.01:
            snap_count += 1

    if verbose and snap_count > 0:
        print(f"  [OBB snap-back] moved {snap_count}/{N} objects closer to original positions")

    boxes[:, 3] = torch.tensor(boxes_np[:, 3], dtype=boxes.dtype, device=device)
    boxes[:, 5] = torch.tensor(boxes_np[:, 5], dtype=boxes.dtype, device=device)
    return boxes
