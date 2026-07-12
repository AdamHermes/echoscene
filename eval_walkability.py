import os
import json
import glob
import os
import argparse
import math
import shapely.geometry
import shapely.ops
import numpy as np

def calculate_walkability(scene_json_path):
    with open(scene_json_path, 'r') as f:
        house = json.load(f)
        
    floor_poly_points = house['rooms'][0]['floorPolygon']
    floor_coords = [(p['x'], p['z']) for p in floor_poly_points]
    floor_poly = shapely.geometry.Polygon(floor_coords)
    
    obj_polys = []
    
    # We reconstruct the object bounds from the scene file
    # We will use the exact bounds given in the JSON
    for obj in house.get('objects_raw', []):
        cx = obj['position']['x']
        cz = obj['position']['z']
        w = obj['size']['x']
        d = obj['size']['z']
        theta = math.radians(obj['rotation']['y'])
        
        hw = w / 2
        hd = d / 2
        local_corners = [(hw, hd), (hw, -hd), (-hw, -hd), (-hw, hd)]
        world_corners = []
        for lx, lz in local_corners:
            rx = lx * math.cos(theta) - lz * math.sin(theta)
            rz = lx * math.sin(theta) + lz * math.cos(theta)
            world_corners.append((cx + rx, cz + rz))
            
        obj_polys.append(shapely.geometry.Polygon(world_corners))
        
    if not obj_polys:
        return 100.0, floor_poly.area, floor_poly.area, 100
        
    union_objs = shapely.ops.unary_union(obj_polys)
    walkable = floor_poly.difference(union_objs)
    
    walkable_area = walkable.area
    total_area = floor_poly.area
    
    score = (walkable_area / total_area) * 100.0 if total_area > 0 else 0
    
    # We don't have grid points anymore, just analytical area
    return score, walkable_area, total_area, int((walkable_area/total_area)*100)

def main():
    parser = argparse.ArgumentParser(description="Evaluate scene walkability analytically.")
    parser.add_argument('--scenes_dir', type=str, required=True, help="Path to the procthor_scenes directory")
    args = parser.parse_args()

    scene_files = [f for f in os.listdir(args.scenes_dir) if f.endswith('.json') and f != 'walkability_results.json']
    print(f"Found {len(scene_files)} scenes to evaluate.")

    results = {}
    total_score = 0
    
    for sf in scene_files:
        scene_name = sf.replace('.json', '')
        print(f"\nEvaluating Walkability for: {scene_name}")
        
        score, walkable, total, points = calculate_walkability(os.path.join(args.scenes_dir, sf))
        
        results[scene_name] = {
            "walkability_score": score,
            "walkable_area": walkable,
            "total_area": total,
            "points_sampled": points
        }
        total_score += score
        print(f"Score: {score:.2f}% ({walkable:.2f}m² / {total:.2f}m²)")

    avg_score = total_score / len(scene_files) if scene_files else 0
    final_output = {
        "summary": {
            "instance_count": len(scene_files),
            "average_walkability": avg_score
        },
        "scenes": results
    }

    out_path = os.path.join(args.scenes_dir, "walkability_results.json")
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nEvaluation complete! Results saved to {out_path}")
    print(f"Average Walkability: {avg_score:.2f}%")

if __name__ == "__main__":
    main()
