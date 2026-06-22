"""
EchoGuidanceOptimizer — lightweight drop-in replacement for CollisionOptimizer.
Implements the same .gradient(x, data, variance, objectness) interface that
GaussianDiffusion._apply_inference_guidance() already calls, but operates
directly on EchoScene's [B, N, D] layout tensor without external dependencies.

PhyScene-style guidance functions:
  phi_coll   : differentiable AABB collision avoidance
  phi_layout : penalize objects outside room boundary (if provided)
"""

from asyncio.log import logger

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class EchoGuidanceOptimizer:
    """
    Stateless guidance optimizer compatible with GaussianDiffusion._apply_inference_guidance().
    
    Expected tensor format (matches _apply_inference_guidance scatter logic):
      x[b, i] = [tx, ty, tz, sx, sy, sz, angle, *class_onehot]
                  0:3  3:6  6:7     7:7+d_class
    
    This is PHYSCENE format (translations first), which is what
    _apply_inference_guidance already builds before calling gradient().
    """

    def __init__(self, cfg, device='cuda', num_classes=16):
        self.device = device
        self.d_bbox = 7          # set externally by echo2layout.py too, kept for safety
        self.d_class = num_classes
        self.dataset = None      # not used, kept for interface compatibility

        # Read weights from cfg, fall back to safe defaults
        self.weight_coll   = float(_cfg_get(cfg, 'weight_coll',   1.0))
        self.weight_layout = float(_cfg_get(cfg, 'weight_layout',  0.0))
        self.clip_val      = float(_cfg_get(cfg, 'clip_grad_by_value', 0.1))
        self.scale         = float(_cfg_get(cfg, 'scale',           1.0))
        self.scale_type    = str  (_cfg_get(cfg, 'scale_type',  'normal'))

        # Optional room boundary box [6] = [min_x, min_y, min_z, max_x, max_y, max_z]
        # Set this externally if you have floor plan info
        self.room_bounds: Optional[torch.Tensor] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface  (called by GaussianDiffusion._apply_inference_guidance)
    # ─────────────────────────────────────────────────────────────────────────

    def gradient(self, x: torch.Tensor, data, variance: torch.Tensor,
                 objectness: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Args:
            x          : [B, N, D]  PHYSCENE-format layout tensor (translations first)
            data       : ignored (kept for interface compatibility)
            variance   : [B, N, 7]  posterior variance, same spatial layout as x[:,:,:7]
            objectness : [B, N, 1]  1 = valid object, 0 = padding

        Returns:
            grad [B, N, D] or None if guidance produces zero loss
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = self._compute_loss(x_in, objectness)
            loss = self._compute_loss(x_in, objectness)
            # logger.debug(
            #     f"[EchoGuidance] loss={'None' if loss is None else f'{loss.item():.6f}'}, "
            #     f"objectness_sum={objectness.sum().item()}, "
            #     f"sizes_mean={x_in[:,:,3:6].mean().item():.4f}, "
            #     f"trans_range=[{x_in[:,:,0:3].min().item():.3f}, {x_in[:,:,0:3].max().item():.3f}]"
            # )
            if loss is None or (isinstance(loss, torch.Tensor) and loss.item() < 1e-8):
                return None

            grad = torch.autograd.grad(loss, x_in)[0]   # [B, N, D]

        # Zero gradients for dimensions we never want to move
        # x format: [tx(0), ty(1), tz(2), sx(3), sy(4), sz(5), angle(6), ...]
        grad[:, :, 1]  = 0.0    # ty (vertical) — furniture sits on floor
        grad[:, :, 3:6] = 0.0   # sx, sy, sz   — don't resize objects
        grad[:, :, 6]  = 0.0    # angle         — don't rotate
        if x.shape[-1] > 7:
            grad[:, :, 7:] = 0.0  # class one-hot — never touch

        # Clip gradient per element
        grad = torch.clamp(grad, -self.clip_val, self.clip_val)

        # Scale by variance (PhyScene Eq. 4: μ + λΣg)
        if variance is not None and self.scale_type == 'normal':
            # variance is [B, N, 7]; only translation dims 0:3 matter after zeroing
            grad[:, :, :7] = self.scale * grad[:, :, :7] * variance[:, :, :7]
        elif self.scale_type == 'div_var':
            grad = self.scale * grad
        # else: use raw clipped grad

        return grad.detach()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal guidance functions
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, x: torch.Tensor,
                      objectness: torch.Tensor) -> Optional[torch.Tensor]:
        """
        x          : [B, N, D]  requires_grad=True
        objectness : [B, N, 1]
        """
        B, N, _ = x.shape
        loss_terms = []

        translations = x[:, :, 0:3]    # [B, N, 3]
        sizes        = x[:, :, 3:6]    # [B, N, 3]  (full sizes, not half)
        valid        = objectness[:, :, 0].bool()   # [B, N]

        # ── 1. Collision avoidance ──────────────────────────────────────────
        if self.weight_coll > 0:
            coll_loss = self._collision_loss(translations, sizes, valid)
            if coll_loss is not None:
                loss_terms.append(self.weight_coll * coll_loss)

        # ── 2. Room layout (optional) ───────────────────────────────────────
        if self.weight_layout > 0 and self.room_bounds is not None:
            layout_loss = self._layout_loss(translations, sizes, valid)
            if layout_loss is not None:
                loss_terms.append(self.weight_layout * layout_loss)

        if len(loss_terms) == 0:
            return None
        return torch.stack(loss_terms).sum()

    def _collision_loss(self, translations, sizes, valid):
        """
        Differentiable AABB penetration loss.
        translations : [B, N, 3]
        sizes        : [B, N, 3]  full extents (not half)
        valid        : [B, N] bool
        Returns scalar or None.
        """
        B, N, _ = translations.shape
        half = sizes.clamp(min=1e-4) * 0.5   # [B, N, 3]

        # Pairwise centre distance and overlap
        # [B, N, 1, 3] - [B, 1, N, 3]
        t_i = translations.unsqueeze(2)   # [B, N, 1, 3]
        t_j = translations.unsqueeze(1)   # [B, 1, N, 3]
        h_i = half.unsqueeze(2)           # [B, N, 1, 3]
        h_j = half.unsqueeze(1)           # [B, 1, N, 3]

        dist = torch.abs(t_i - t_j)                      # [B, N, N, 3]
        overlap = torch.relu(h_i + h_j - dist)            # [B, N, N, 3]  per-axis
        penetration_vol = overlap[..., 0] * overlap[..., 1] * overlap[..., 2]  # [B, N, N]

        # Mask: only valid pairs, exclude diagonal
        valid_i = valid.unsqueeze(2).float()   # [B, N, 1]
        valid_j = valid.unsqueeze(1).float()   # [B, 1, N]
        pair_mask = valid_i * valid_j          # [B, N, N]
        diag = torch.eye(N, device=translations.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - diag)

        # Upper-triangle only to avoid double-counting
        triu = torch.triu(torch.ones(N, N, device=translations.device), diagonal=1)
        pair_mask = pair_mask * triu.unsqueeze(0)

        masked_pen = penetration_vol * pair_mask
        n_pairs = pair_mask.sum().clamp(min=1.0)

        if masked_pen.sum().item() == 0.0:
            return None
        return masked_pen.sum() / n_pairs

    def _layout_loss(self, translations, sizes, valid):
        """
        Penalize objects whose centres fall outside room_bounds.
        room_bounds: [6] = [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        bounds = self.room_bounds.to(translations.device)
        mins = bounds[:3].unsqueeze(0).unsqueeze(0)   # [1,1,3]
        maxs = bounds[3:].unsqueeze(0).unsqueeze(0)

        # Penalty = amount by which centre exceeds bounds
        below = torch.relu(mins - translations)   # [B, N, 3]
        above = torch.relu(translations - maxs)
        violation = (below + above).sum(dim=-1)   # [B, N]

        valid_f = valid.float()
        n_valid = valid_f.sum().clamp(min=1.0)
        return (violation * valid_f).sum() / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, '__getitem__'):
        try:
            return cfg[key]
        except (KeyError, TypeError):
            pass
    return getattr(cfg, key, default)