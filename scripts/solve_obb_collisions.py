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
    
    # Half extents on X and Z
    hx, hz = dx / 2, dz / 2
    
    # Local corners
    corners_local = [
        np.array([hx, hz]),
        np.array([-hx, hz]),
        np.array([-hx, -hz]),
        np.array([hx, -hz])
    ]
    
    # Rotate and translate
    corners = []
    for cx, cz in corners_local:
        # standard 2D rotation
        nx = cx * c - cz * s + x
        nz = cx * s + cz * c + z
        corners.append(np.array([nx, nz]))
        
    return corners

def get_axes(corners):
    axes = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        edge = p2 - p1
        # Normal
        normal = np.array([-edge[1], edge[0]])
        length = np.linalg.norm(normal)
        if length > 0:
            axes.append(normal / length)
    return axes[:2]

def project(corners, axis):
    dots = [np.dot(c, axis) for c in corners]
    return min(dots), max(dots)

def sat_intersection(cornersA, cornersB):
    axes = get_axes(cornersA) + get_axes(cornersB)
    min_overlap = float('inf')
    mtv = None
    
    for axis in axes:
        minA, maxA = project(cornersA, axis)
        minB, maxB = project(cornersB, axis)
        
        if maxA < minB or maxB < minA:
            return False, None
            
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            
            centerA = np.mean(cornersA, axis=0)
            centerB = np.mean(cornersB, axis=0)
            if np.dot(centerA - centerB, axis) < 0:
                mtv = -axis * overlap
            else:
                mtv = axis * overlap
                
    return True, mtv

def check_y_overlap(tA, sA, tB, sB):
    # Using Y as the height
    yA = tA[1]
    dyA = sA[1]
    yB = tB[1]
    dyB = sB[1]
    
    minA = yA - dyA / 2
    maxA = yA + dyA / 2
    minB = yB - dyB / 2
    maxB = yB + dyB / 2
    
    return not (maxA < minB or maxB < minA)

def solve_collisions():
    input_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/physcene_collision_input_merged.json'
    output_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/physcene_collision_resolved.json'
    
    print(f"Loading {input_file} ...")
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    num_scenes = len(data['scene_ids'])
    
    for s_idx in range(num_scenes):
        translations = np.array(data['translations'][s_idx])
        sizes = data['sizes'][s_idx]
        angles = data['angles'][s_idx]
        
        num_objs = len(translations)
        
        iterations = 50
        for it in range(iterations):
            collision_found = False
            for i in range(num_objs):
                for j in range(i + 1, num_objs):
                    tA = translations[i]
                    sA = sizes[i]
                    aA = angles[i]
                    
                    tB = translations[j]
                    sB = sizes[j]
                    aB = angles[j]
                    
                    if not check_y_overlap(tA, sA, tB, sB):
                        continue
                        
                    cornersA = get_corners(tA, sA, aA)
                    cornersB = get_corners(tB, sB, aB)
                    
                    colliding, mtv = sat_intersection(cornersA, cornersB)
                    if colliding:
                        collision_found = True
                        # Push them apart symmetrically on X and Z
                        # translations is [x, y, z], so index 0 is x, index 2 is z
                        translations[i][0] += mtv[0] / 2
                        translations[i][2] += mtv[1] / 2
                        translations[j][0] -= mtv[0] / 2
                        translations[j][2] -= mtv[1] / 2
            
            if not collision_found:
                break
                
        data['translations'][s_idx] = translations.tolist()
        print(f"Scene {s_idx}/{num_scenes} ({data['scene_ids'][s_idx]}) resolved in {it+1} iterations.")
        
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
    print(f"\nResolved collisions saved to {output_file}")

if __name__ == '__main__':
    solve_collisions()
