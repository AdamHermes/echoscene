import os
import json
import numpy as np
import shapely.geometry
import shapely.ops
import argparse

def calculate_walkability_2d(scene_json_path, agent_radius=0.2):
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)
        
    floor_poly = scene["rooms"][0]["floorPolygon"]
    room_shape = shapely.geometry.Polygon([(p['x'], p['z']) for p in floor_poly])
    total_area = room_shape.area
    
    obj_polys = []
    if "objects_raw" in scene:
        for obj in scene["objects_raw"]:
            x, z = obj['position']['x'], obj['position']['z']
            sx, sz = obj['size']['x'], obj['size']['z']
            theta = np.radians(obj['rotation']['y'])
            hw, hd = sx/2.0, sz/2.0
            corners = [(hw, hd), (hw, -hd), (-hw, -hd), (-hw, hd)]
            world_corners = []
            for lx, lz in corners:
                rx = lx * np.cos(theta) - lz * np.sin(theta)
                rz = lx * np.sin(theta) + lz * np.cos(theta)
                world_corners.append((x + rx, z + rz))
            poly = shapely.geometry.Polygon(world_corners)
            # Expand the object by agent_radius
            if agent_radius > 0:
                poly = poly.buffer(agent_radius, resolution=4)
            obj_polys.append(poly)
            
    # Shrink the room by agent_radius (walls)
    if agent_radius > 0:
        walkable_shape = room_shape.buffer(-agent_radius, resolution=4)
    else:
        walkable_shape = room_shape

    if not walkable_shape.is_empty:
        union_objs = shapely.ops.unary_union(obj_polys)
        walkable_shape = walkable_shape.difference(union_objs)
        walkable_area = walkable_shape.area
    else:
        walkable_area = 0.0
    
    return walkable_area / total_area if total_area > 0 else 0, walkable_area, total_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_dir", type=str, required=True)
    parser.add_argument("--agent_radius", type=float, default=0.2)
    args = parser.parse_args()
    
    scores = []
    for f in os.listdir(args.scenes_dir):
        if f.endswith('.json') and not f.endswith('results.json'):
            path = os.path.join(args.scenes_dir, f)
            score, w, t = calculate_walkability_2d(path, args.agent_radius)
            scores.append(score)
            # print(f"{f}: {score:.2%} ({w:.2f}/{t:.2f})")
    
    avg = np.mean(scores)
    print(f"\nAverage Walkability (Radius={args.agent_radius}): {avg:.2%}")
