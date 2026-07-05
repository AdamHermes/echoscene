import torch

'''
 https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py
'''
from .oriented_iou_loss import cal_iou_3d


def _to_zup_convention(sizes, centers, angle_rad):
    """
    EchoScene convention: y is vertical, rotation is yaw about y (footprint in x-z plane).
    oriented_iou_loss convention: z is vertical, rotation is about z (footprint in x-y plane).
    Remap: their(x,y) = our(x,z); their(z) = our(y); their(w,h) = our(sx,sz); their(l) = our(sy).
    """
    x, y_, z = centers[..., 0], centers[..., 1], centers[..., 2]
    sx, sy, sz = sizes[..., 0], sizes[..., 1], sizes[..., 2]
    return torch.stack([x, z, y_, sx, sz, sy, angle_rad], dim=-1)  # (..., 7)


def oriented_pairwise_collision_loss(
    boxes,            # (N, 7): [sx, sy, sz, tx, ty, tz, angle_rad] — already denormalized & arctan'd
    scene_ids=None,
    valid_mask=None,
    reduction='mean',
):
    N = boxes.shape[0]
    if N < 2:
        return boxes.new_zeros(())

    sizes = boxes[:, 0:3]
    centers = boxes[:, 3:6]
    angle = boxes[:, 6]   # already radians — _denormalize_box_params already ran postprocess_sincos2arctan

    box7 = _to_zup_convention(sizes, centers, angle)  # (N, 7)

    pair_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=boxes.device), diagonal=1)
    if scene_ids is not None:
        scene_ids = scene_ids.to(device=boxes.device, dtype=torch.long)
        pair_mask &= (scene_ids[:, None] == scene_ids[None, :])
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=boxes.device, dtype=torch.bool)
        pair_mask &= (valid_mask[:, None] & valid_mask[None, :])

    idx_i, idx_j = pair_mask.nonzero(as_tuple=True)
    if idx_i.numel() == 0:
        return boxes.new_zeros(())

    box_i = box7[idx_i].unsqueeze(0)
    box_j = box7[idx_j].unsqueeze(0)

    iou3d = cal_iou_3d(box_i, box_j).squeeze(0)

    if reduction == 'sum':
        return iou3d.sum()
    if reduction == 'none':
        return iou3d
    return iou3d.mean()

def differentiable_aabb_collision_loss(
    boxes,
    scene_ids=None,
    reduction='mean',
    eps=1e-6,
):
    """Differentiable pairwise AABB penetration volume loss.

    boxes is expected in [size_x, size_y, size_z, center_x, center_y, center_z, ...]
    format. Extra dimensions such as angle are ignored.
    """
    if boxes.numel() == 0 or boxes.shape[0] < 2:
        return boxes.new_zeros(())

    sizes = boxes[:, :3].clamp(min=eps)
    centers = boxes[:, 3:6]
    half_sizes = sizes * 0.5
    overlap = torch.relu(
        half_sizes[:, None, :] + half_sizes[None, :, :] -
        torch.abs(centers[:, None, :] - centers[None, :, :])
    )
    pair_loss = overlap[..., 0] * overlap[..., 1] * overlap[..., 2]

    pair_mask = torch.triu(
        torch.ones(pair_loss.shape, dtype=torch.bool, device=pair_loss.device),
        diagonal=1,
    )
    if scene_ids is not None:
        if not isinstance(scene_ids, torch.Tensor):
            scene_ids = torch.as_tensor(scene_ids, dtype=torch.long, device=boxes.device)
        else:
            scene_ids = scene_ids.to(device=boxes.device, dtype=torch.long)
        pair_mask = pair_mask & (scene_ids[:, None] == scene_ids[None, :])

    valid_losses = pair_loss[pair_mask]
    if valid_losses.numel() == 0:
        return boxes.new_zeros(())
    if reduction == 'sum':
        return valid_losses.sum()
    if reduction == 'none':
        return valid_losses
    return valid_losses.mean()

def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
        ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
        of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
        bboxes1 and bboxes2.
        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Defaults to 1e-6.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
