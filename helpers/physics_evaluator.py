import torch


def _as_scene_ids(scene_ids, num_boxes, device):
    if scene_ids is None:
        return torch.zeros(num_boxes, dtype=torch.long, device=device)
    if isinstance(scene_ids, torch.Tensor):
        return scene_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(scene_ids, dtype=torch.long, device=device)


def _pairwise_penetration(boxes):
    sizes = boxes[:, :3].clamp(min=1e-6)
    centers = boxes[:, 3:6]
    half = sizes * 0.5
    center_delta = torch.abs(centers[:, None, :] - centers[None, :, :])
    overlap = torch.relu(half[:, None, :] + half[None, :, :] - center_delta)
    volume = overlap[..., 0] * overlap[..., 1] * overlap[..., 2]
    depth = overlap.max(dim=-1).values
    return volume, depth


def evaluate_layout_physics(boxes, scene_ids=None, floor_y=0.0, contact_eps=0.03,
                            penetration_eps=1e-6):
    """
    Evaluate denormalized EchoScene boxes in [l, h, w, x, y, z] format.

    ColObj is the fraction of colliding object pairs. ColScene is the fraction
    of scenes with at least one colliding pair.
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    boxes = boxes.float()
    device = boxes.device
    scene_ids = _as_scene_ids(scene_ids, boxes.shape[0], device)

    total_pairs = 0
    colliding_pairs = 0
    scenes_with_collision = 0
    scene_count = 0
    penetration_means = []
    penetration_maxes = []

    for scene_id in torch.unique(scene_ids):
        scene_boxes = boxes[scene_ids == scene_id]
        num_boxes = int(scene_boxes.shape[0])
        if num_boxes == 0:
            continue
        scene_count += 1
        if num_boxes < 2:
            continue

        penetration, depth = _pairwise_penetration(scene_boxes)
        pair_mask = torch.triu(
            torch.ones(num_boxes, num_boxes, dtype=torch.bool, device=device),
            diagonal=1,
        )
        pair_penetration = penetration[pair_mask]
        pair_depth = depth[pair_mask]
        pair_collisions = pair_penetration > penetration_eps

        total_pairs += int(pair_penetration.numel())
        num_collisions = int(pair_collisions.sum().item())
        colliding_pairs += num_collisions
        if num_collisions > 0:
            scenes_with_collision += 1
        if pair_penetration.numel() > 0:
            penetration_means.append(pair_penetration.mean())
            penetration_maxes.append(pair_depth.max())

    bottoms = boxes[:, 4] - boxes[:, 1].clamp(min=1e-6) * 0.5
    floor = torch.as_tensor(floor_y, dtype=boxes.dtype, device=device)
    grounding_error = torch.relu(floor - bottoms)
    floating_error = torch.relu(bottoms - (floor + contact_eps))

    return {
        "col_obj": colliding_pairs / max(total_pairs, 1),
        "col_scene": scenes_with_collision / max(scene_count, 1),
        "num_pairs": total_pairs,
        "colliding_pairs": colliding_pairs,
        "num_scenes": scene_count,
        "scenes_with_collision": scenes_with_collision,
        "avg_pair_penetration": float(torch.stack(penetration_means).mean().item()) if penetration_means else 0.0,
        "max_pair_penetration": float(torch.stack(penetration_maxes).max().item()) if penetration_maxes else 0.0,
        "grounding_error": float(grounding_error.mean().item()) if boxes.numel() else 0.0,
        "floating_rate": float((floating_error > contact_eps).float().mean().item()) if boxes.numel() else 0.0,
    }
