import torch
import numpy as np
import cv2
import heapq
from .loss import axis_aligned_bbox_overlaps_3d
from .oriented_iou_loss import cal_iou_3d

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

def calc_loss_on_path(image, shortest_path, robot_width, robot_width_real, robot_hight_real, map_to_image_coordinate, image_to_map_coordinate, scale, image_size, bbox_floor):
    loss_walkable = 0.0
    box_mask = image[:,:,1] == 255
    bbox_path = []
    path_count = 0
    for i in range(len(shortest_path)):
        if box_mask[shortest_path[i]]:
            if path_count % robot_width == 0:
                center_map = image_to_map_coordinate((shortest_path[i][1], shortest_path[i][0]))
                angle = 0.
                # cal_iou_3d expects (X, Z, Y_up, X_size, Z_size, Y_size_up, alpha)
                box = np.array([center_map[0], center_map[1], 0.0, robot_width_real, robot_width_real, robot_hight_real, angle])
                bbox_path.append(box)
            path_count += 1
            
    if not bbox_path:
        return loss_walkable
        
    for box in bbox_path:
        center = map_to_image_coordinate((box[0], box[2]))
        size = (int(box[5] / scale * image_size / 2), int(box[3] / scale * image_size / 2)) # l and w
        angle = box[-1]
        box_points = cv2.boxPoints(((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)
        cv2.drawContours(image, [box_points], 0, (0, 255, 255), robot_width)
        
    bbox_path = np.expand_dims(np.stack(bbox_path, 0), 0)
    bbox_cnt_path = bbox_path.shape[1]
    bbox_floor_exp = bbox_floor.unsqueeze(0) if bbox_floor.dim() == 2 else bbox_floor
    bbox_floor_cur_cnt = bbox_floor_exp.shape[1]
    
    for bbox_cnt_idx in range(bbox_floor_cur_cnt):    
        bbox_target = bbox_floor_exp[:, bbox_cnt_idx, :]
        bbox_target = torch.tile(bbox_target.unsqueeze(1), [1, bbox_cnt_path, 1])
        loss_walkable = loss_walkable + cal_iou_3d(
            torch.tensor(bbox_path, device=bbox_target.device, dtype=bbox_target.dtype), 
            bbox_target
        ).sum() / max(1, len(bbox_floor)) / bbox_floor_cur_cnt
        
    return loss_walkable

def compute_walkable_loss(bbox, floor_plan, objectness=None, robot_width_real=0.5, robot_hight_real=1.5):
    """
    Computes Reachability Guidance by verifying an agent can traverse the room.
    """
    if floor_plan is None:
        if len(bbox.shape) == 2:
            bbox = bbox.unsqueeze(0)
        centers_obj = bbox[:, :, 3:6]
        dist_sq = centers_obj[:, :, 0]**2 + centers_obj[:, :, 2]**2
        sigma = 0.5
        walk_penalty = torch.exp(-dist_sq / sigma).sum()
        return walk_penalty

    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)
        if objectness is not None and len(objectness.shape) == 1:
            objectness = objectness.unsqueeze(0)
        
    loss_walkable = 0.0
    for i in range(len(bbox)):
        bbox_cur = bbox[i:i+1, :, :]
        if objectness is not None:
            obj_mask = objectness[i]
            if obj_mask.dim() > 1:
                obj_mask = obj_mask[:, 0]
            bbox_cur = bbox_cur[:, obj_mask.bool(), :]
        
        bbox_cur_cnt = bbox_cur.shape[1]
        
        if isinstance(floor_plan, list) and len(floor_plan) > i:
            fp = floor_plan[i]
        else:
            fp = floor_plan
            
        if fp is None or len(fp) != 2:
            continue
            
        vertices, faces = fp
        
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
            
        vertices = vertices - np.mean(vertices, axis=0)
        vertices = vertices[:, 0::2]
        scale = np.abs(vertices).max() + 0.2
        
        bbox_floor = bbox_cur[0, bbox_cur[0, :, 4] < robot_hight_real]
        
        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        robot_width = int(robot_width_real / scale * image_size/2)

        def map_to_image_coordinate(point):
            x, y = point
            x_image = int(x / scale * image_size/2)+image_size/2
            y_image = int(y / scale * image_size/2)+image_size/2
            return x_image, y_image
        
        def image_to_map_coordinate(point):
            x, y = point
            x_map = (x - image_size/2) * 2 / image_size *scale
            y_map = (y - image_size/2) * 2 / image_size *scale
            return x_map, y_map

        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [map_to_image_coordinate(v) for v in face_vertices]
            pts = np.array(face_vertices_image, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(image, [pts], (255, 0, 0))

        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        floor_plan_mask = image[:, :, 0] == 255
        box_heat_map = np.zeros((image_size, image_size), dtype=np.float32)

        for box in bbox_floor:
            box = box.cpu().detach().numpy()
            center = map_to_image_coordinate((box[3], box[5]))
            size = (int(box[0] / scale * image_size / 2), int(box[2] / scale * image_size / 2))
            angle = box[-1]

            box_points = cv2.boxPoints(((center[0], center[1]), size, -angle/np.pi*180))
            box_points = np.intp(box_points)
            
            box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.drawContours(image, [box_points], 0, (0, 255, 0), robot_width)
            cv2.fillPoly(image, [box_points], (0, 255, 0))
            cv2.drawContours(box_mask, [box_points], 0, (0, 255, 0), robot_width)
            cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
            
            box_mask_bool = box_mask[:,:,1] == 255
            if min(size) != 0:
                gaussian = draw_2d_gaussian((int(center[0]), int(center[1])), size, -angle, image_size)
                box_heat_map = box_heat_map + gaussian * box_mask_bool
            
        box_heat_map = floor_plan_mask * box_heat_map
        box_wall_heat_map = box_heat_map + (1 - floor_plan_mask) * box_heat_map.max()
        
        walkable_map = image[:, :, 0].copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_map, connectivity=8)
        
        if num_labels > 2:
            area_1 = np.zeros_like(walkable_map)
            area_2 = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 1
                if mask.sum() > area_2.sum():
                    area_2 = mask.copy()
                if area_2.sum() > area_1.sum():
                    area_2, area_1 = area_1.copy(), area_2.copy()
                    
            if area_2.sum() > 100:
                dist_tf_1 = cv2.distanceTransform(area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                minimum_area_1 = np.argmax(dist_tf_1)
                minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
                
                dist_tf_2 = cv2.distanceTransform(area_2.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                minimum_area_2 = np.argmax(dist_tf_2)
                minimum_area_2_position = np.unravel_index(minimum_area_2, area_2.shape)
                
                shortest_path = find_shortest_path(
                    box_wall_heat_map, 
                    (minimum_area_1_position[0], minimum_area_1_position[1]),
                    (minimum_area_2_position[0], minimum_area_2_position[1])
                )
                if shortest_path is not None:
                    mapped_bbox_floor = bbox_floor[:, [3, 5, 4, 0, 2, 1, 6]]
                    loss_walkable = loss_walkable + calc_loss_on_path(
                        image, shortest_path, robot_width, robot_width_real, robot_hight_real,
                        map_to_image_coordinate, image_to_map_coordinate,
                        scale, image_size, mapped_bbox_floor
                    )

    return loss_walkable
