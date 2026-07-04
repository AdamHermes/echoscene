import torch
import numpy as np
import cv2
import heapq
from .loss import axis_aligned_bbox_overlaps_3d

def draw_2d_gaussian(center, size, angle, image_size = 256):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    covariance_matrix = np.array([
        [size[0]**2, 0],
        [0, size[1]**2]
    ])
    rotation_convariance_matrix = rotation_matrix @ covariance_matrix @ rotation_matrix.T

    x = np.arange(0,image_size)
    y = np.arange(0,image_size)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()]).T -center
    try:
        z = np.sum((xy @ np.linalg.inv(rotation_convariance_matrix)) * xy, axis=1)
    except:
        z = np.zeros(xx.shape[0]*xx.shape[1])
    gaussian = np.exp(-0.5 * z)
    gaussian = gaussian.reshape(xx.shape)
    return gaussian

def heuristic_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_shortest_path(matrix, start, end):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = [(0, start)]
    parent_map = {}
    g_cost = {node: float('inf') for node in np.ndindex(matrix.shape)}
    g_cost[start] = 0
    count = 0
    while open_set and count<5000:
        count+=1
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in parent_map:
                path.append(current)
                current = parent_map[current]
            path.append(start)
            return path[::-1]

        for direction in directions:
            new_node = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= new_node[0] < matrix.shape[0] and 0 <= new_node[1] < matrix.shape[1]:
                tentative_g_cost = g_cost[current] + matrix[new_node]
                if tentative_g_cost < g_cost[new_node]:
                    parent_map[new_node] = current
                    g_cost[new_node] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic_distance(new_node, end) * 0.01
                    heapq.heappush(open_set, (f_cost, new_node))

    return None

def compute_room_outer_loss(bbox, room_outer_box=None, scene_ids=None, objectness=None):
    """
    Computes Room-Layout Guidance loss by penalizing overlap with infinite walls or boundaries.
    If room_outer_box is None, we attempt to use the floor bounding box (found via ~objectness)
    as the boundary. If that fails, we fallback to a hardcoded 6m x 6m bounds: [-3, 3] in X/Z.
    """
    # Note: bbox format is [N, 7] (sizes, translations, angle) because we are passing it in directly.
    # Actually _denormalize_box_params returns [sizes, translations, angle]. 
    # Let's extract the sizes and centers.
    if len(bbox.shape) == 3:
        # If batched, flatten it or just take the first batch if B=1
        bbox = bbox.squeeze(0)
        
    half_sizes_obj = bbox[:, :3].clamp(min=1e-4) * 0.5
    centers_obj = bbox[:, 3:6]
    
    max_corners = centers_obj + half_sizes_obj
    min_corners = centers_obj - half_sizes_obj
    
    total_loss = 0.0
    
    unique_scenes = torch.unique(scene_ids) if scene_ids is not None else [0]
    
    for scene_id in unique_scenes:
        if scene_ids is not None:
            scene_mask = (scene_ids == scene_id)
        else:
            scene_mask = torch.ones(bbox.shape[0], dtype=torch.bool, device=bbox.device)
            
        if objectness is not None:
            obj_mask = scene_mask & objectness.to(dtype=torch.bool, device=bbox.device)
            non_obj_mask = scene_mask & ~objectness.to(dtype=torch.bool, device=bbox.device)
        else:
            obj_mask = scene_mask
            non_obj_mask = torch.zeros_like(scene_mask)
            
        if not obj_mask.any():
            continue
            
        # Default fallback boundaries
        max_bound_x = 3.0
        min_bound_x = -3.0
        max_bound_z = 3.0
        min_bound_z = -3.0
        
        # Extract the floor boundary if available
        if non_obj_mask.any():
            # non_obj_mask contains background objects like `_scene_` and `floor`.
            # Find the largest object (by X * Z area) which is typically the floor.
            non_obj_idx = torch.where(non_obj_mask)[0]
            sizes_x = half_sizes_obj[non_obj_idx, 0]
            sizes_z = half_sizes_obj[non_obj_idx, 2]
            areas = sizes_x * sizes_z
            best_idx = non_obj_idx[torch.argmax(areas)]
            
            best_center = centers_obj[best_idx].detach()
            best_half_size = half_sizes_obj[best_idx].detach()
            
            max_bound_x = best_center[0] + best_half_size[0]
            min_bound_x = best_center[0] - best_half_size[0]
            max_bound_z = best_center[2] + best_half_size[2]
            min_bound_z = best_center[2] - best_half_size[2]
        else:
            print("Warning: No floor object found for scene. Falling back to default [-3.0, 3.0] boundaries for room outer loss.")
            
        # Compute L1 penalty for objects exceeding these boundaries
        cur_max_corners = max_corners[obj_mask]
        cur_min_corners = min_corners[obj_mask]
        
        loss_x_max = torch.relu(cur_max_corners[:, 0] - max_bound_x).sum()
        loss_x_min = torch.relu(min_bound_x - cur_min_corners[:, 0]).sum()
        loss_z_max = torch.relu(cur_max_corners[:, 2] - max_bound_z).sum()
        loss_z_min = torch.relu(min_bound_z - cur_min_corners[:, 2]).sum()
        
        total_loss = total_loss + loss_x_max + loss_x_min + loss_z_max + loss_z_min
        
    return total_loss

def compute_walkable_loss(bbox, floor_plan, robot_width_real=0.5, robot_hight_real=1.5):
    """
    Computes Reachability Guidance by verifying an agent can traverse the room.
    If floor_plan is None, we apply a soft penalty to objects placed directly in the center (0,0)
    to enforce a walkable central area.
    """
    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)
    centers_obj = bbox[:, :, 3:6]
    
    # Penalize objects based on Gaussian distance to origin (X=0, Z=0)
    # Centers close to (0,0) will have a higher penalty.
    dist_sq = centers_obj[:, :, 0]**2 + centers_obj[:, :, 2]**2
    # The penalty decays as objects move further from the center. 
    # sigma = 0.5 means strong penalty within ~0.7 meters from center.
    sigma = 0.5
    walk_penalty = torch.exp(-dist_sq / sigma).sum()
    
    return walk_penalty
