import json
import numpy as np
import math

def get_corners(translation, size, angle):
    x, y, z = translation
    dx, dy, dz = size
    theta = angle[0]
    
    # 2D corners on the XZ plane relative to center
    c = math.cos(theta)
    s = math.sin(theta)
    
    hx, hz = dx / 2, dz / 2
    
    corners_local = [
        np.array([hx, hz]),
        np.array([-hx, hz]),
        np.array([-hx, -hz]),
        np.array([hx, -hz])
    ]
    
    corners = []
    for cx, cz in corners_local:
        # Rotate around Y axis and translate
        nx = cx * c - cz * s + x
        nz = cx * s + cz * c + z
        corners.append(np.array([nx, nz]))
        
    return corners

def get_axes(corners):
    axes = []
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        edge = p1 - p2
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            axes.append(normal)
    # The above gives 4 axes, but for rectangles opposite sides are parallel.
    # To avoid double correcting containment, we just return the first 2 unique axes.
    return [axes[0], axes[1]]

def project(corners, axis):
    dots = [np.dot(c, axis) for c in corners]
    return min(dots), max(dots)

def overlap(minA, maxA, minB, maxB):
    if minA < minB:
        return maxA > minB, maxA - minB
    else:
        return maxB > minA, maxB - minA

def check_intersection_and_mtv(cornersA, cornersB):
    axes = get_axes(cornersA) + get_axes(cornersB)
    
    min_overlap = float('inf')
    mtv_axis = None
    
    for axis in axes:
        minA, maxA = project(cornersA, axis)
        minB, maxB = project(cornersB, axis)
        
        intersect, o_lap = overlap(minA, maxA, minB, maxB)
        if not intersect:
            return False, None
            
        if o_lap < min_overlap:
            min_overlap = o_lap
            mtv_axis = axis
            
    # Direction vector from center A to center B
    centerA = np.mean(cornersA, axis=0)
    centerB = np.mean(cornersB, axis=0)
    dir_vec = centerB - centerA
    
    if np.dot(dir_vec, mtv_axis) < 0:
        mtv_axis = -mtv_axis
        
    mtv = mtv_axis * min_overlap
    return True, mtv

def check_containment_and_mtv(cornersA, cornersL):
    axes = get_axes(cornersL)
    mtv = np.zeros(2)
    for axis in axes:
        minA, maxA = project(cornersA, axis)
        minL, maxL = project(cornersL, axis)
        
        if minA < minL:
            mtv += axis * (minL - minA)
        elif maxA > maxL:
            mtv += axis * (maxL - maxA)
            
    return np.linalg.norm(mtv) > 1e-5, mtv

def resolve_collisions(translations, sizes, angles, labels, objectness):
    # Translations are lists [x, y, z]
    n = len(translations)
    resolved = [np.array([t[0], t[2]]) for t in translations] # Extract (x, z)
    
    # Determine which objects should be ignored (floor, layout, lamps)
    ignore = []
    layout_indices = []
    for i in range(n):
        idx = -1
        if labels and i < len(labels) and sum(labels[i]) > 0:
            idx = int(np.argmax(labels[i]))
            
        obj_mask = 1.0
        if objectness and i < len(objectness):
            obj_mask = objectness[i][0]
            
        # Ignore Floor/Layout (idx 14, 6, 0 or low objectness) and Lamps (idx 7)
        is_layout = (obj_mask < 0.5) or (idx in [0, 6, 14])
        is_lamp = (idx == 7)
        if is_layout:
            ignore.append(i)
            layout_indices.append(i)
        elif is_lamp:
            ignore.append(i)
    
    # We run iterative resolution
    for _ in range(500):
        collision_found = False
        
        # 1. Enforce containment within layout(s)
        for L_idx in layout_indices:
            cL = get_corners([resolved[L_idx][0], translations[L_idx][1], resolved[L_idx][1]], sizes[L_idx], angles[L_idx])
            for i in range(n):
                if i in ignore:
                    continue
                cA = get_corners([resolved[i][0], translations[i][1], resolved[i][1]], sizes[i], angles[i])
                violated, mtv = check_containment_and_mtv(cA, cL)
                if violated:
                    resolved[i] += mtv
                    collision_found = True
                    
        # 2. Resolve object-object overlaps
        for i in range(n):
            if i in ignore:
                continue
                
            for j in range(i + 1, n):
                if j in ignore:
                    continue
                    
                tA = [resolved[i][0], translations[i][1], resolved[i][1]]
                tB = [resolved[j][0], translations[j][1], resolved[j][1]]
                
                sA = sizes[i]
                sB = sizes[j]
                
                # Check vertical bounds (Y axis)
                minYA = tA[1] - sA[1]/2
                maxYA = tA[1] + sA[1]/2
                minYB = tB[1] - sB[1]/2
                maxYB = tB[1] + sB[1]/2
                
                if maxYA <= minYB or maxYB <= minYA:
                    continue # No vertical intersection
                
                cA = get_corners(tA, sA, angles[i])
                cB = get_corners(tB, sB, angles[j])
                
                intersect, mtv = check_intersection_and_mtv(cA, cB)
                if intersect:
                    collision_found = True
                    # Push objects apart symmetrically
                    resolved[i] -= mtv * 0.5
                    resolved[j] += mtv * 0.5
                    
        if not collision_found:
            break
            
    final_translations = []
    for i in range(n):
        final_translations.append([resolved[i][0], translations[i][1], resolved[i][1]])
        
    return final_translations

def main():
    input_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input_sorted.json'
    output_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json'
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    num_scenes = len(data['scene_ids'])
    
    # Create output dict maintaining original data but with updated translations
    out_data = {k: v for k, v in data.items()}
    out_data['translations'] = []
    
    print(f"Resolving collisions for {num_scenes} scenes on the XZ plane...")
    for idx in range(num_scenes):
        translations = data['translations'][idx]
        sizes = data['sizes'][idx]
        angles = data['angles'][idx]
        labels = data.get('class_labels', [[] for _ in range(num_scenes)])[idx]
        objectness = data.get('objectness', [[] for _ in range(num_scenes)])[idx]
        
        resolved_t = resolve_collisions(translations, sizes, angles, labels, objectness)
        out_data['translations'].append(resolved_t)
        
        # print progress
        if idx % 50 == 0:
            print(f"Scene {idx}/{num_scenes-1} resolved.")
            
    with open(output_file, 'w') as f:
        json.dump(out_data, f)
        
    print(f"\\nResolved collisions saved to {output_file}")

if __name__ == '__main__':
    main()
