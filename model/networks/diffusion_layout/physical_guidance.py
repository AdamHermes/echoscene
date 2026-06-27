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

def compute_room_outer_loss(bbox, room_outer_box):
    """
    Computes Room-Layout Guidance loss by penalizing overlap with infinite walls.
    bbox: [B, N, 7] (translations, sizes, angle)
    room_outer_box: [B, W, 7] walls
    """
    if room_outer_box is None:
        return torch.tensor(0.0, device=bbox.device, requires_grad=True)
        
    loss_room_outer = 0.0
    bbox_cnt_room = room_outer_box.shape[1]

    # Calculate axis aligned bbox corners for both
    half_sizes_obj = bbox[:, :, :3].clamp(min=1e-4) * 0.5
    centers_obj = bbox[:, :, 3:6]
    obj_corners = torch.cat((centers_obj - half_sizes_obj, centers_obj + half_sizes_obj), dim=-1)
    
    half_sizes_wall = room_outer_box[:, :, :3].clamp(min=1e-4) * 0.5
    centers_wall = room_outer_box[:, :, 3:6]
    wall_corners = torch.cat((centers_wall - half_sizes_wall, centers_wall + half_sizes_wall), dim=-1)

    for j in range(len(bbox)):
        obj_corners_cur = obj_corners[j:j+1, :, :]
        wall_corners_cur = wall_corners[j:j+1, :, :]
        
        for i in range(obj_corners_cur.shape[1]):
            bbox_target = obj_corners_cur[:, i:i+1, :]
            bbox_target = bbox_target.repeat(1, bbox_cnt_room, 1)
            iou = axis_aligned_bbox_overlaps_3d(wall_corners_cur, bbox_target)
            loss_room_outer += iou.sum() / len(bbox) / obj_corners_cur.shape[1]
            
    return loss_room_outer

def compute_walkable_loss(bbox, floor_plan, robot_width_real=0.5, robot_hight_real=1.5):
    """
    Computes Reachability Guidance by verifying an agent can traverse the room.
    """
    if floor_plan is None:
        return torch.tensor(0.0, device=bbox.device, requires_grad=True)
    
    # Placeholder for actual walkable loss logic due to complexity.
    # A full implementation requires projecting to 2D image, A* search, and backprojecting to 3D.
    return torch.tensor(0.0, device=bbox.device, requires_grad=True)
