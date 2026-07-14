import os
import json
import math
import argparse
import numpy as np

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not found. Please install it.")
    exit(1)

def is_clear(x, z, furniture_boxes, l, w):
    clearance = 0.3
    for box in furniture_boxes:
        min_x, max_x, min_z, max_z = box
        if (min_x - clearance <= x <= max_x + clearance) and (min_z - clearance <= z <= max_z + clearance):
            return False
    if x < clearance or x > l - clearance or z < clearance or z > w - clearance:
        return False
    return True

def evaluate_navigation(scenes_dir):
    json_files = [f for f in os.listdir(scenes_dir) if f.endswith('.json') and not f.startswith('walkability_') and not f.startswith('navigation_')]
    
    if not json_files:
        print(f"No JSON scene files found in {scenes_dir}")
        return

    controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)
    
    results = {}
    total_objects_evaluated = 0
    total_objects_accessible = 0

    print(f"Found {len(json_files)} scenes to evaluate for Navigation Accessibility.")
    
    for filename in json_files:
        scene_name = filename.replace('.json', '')
        file_path = os.path.join(scenes_dir, filename)
        
        with open(file_path, 'r') as f:
            house_data = json.load(f)
            
        rd = house_data.get('room_dims', {})
        l = rd.get('l', 0)
        w = rd.get('w', 0)
        furniture_boxes = house_data.get('furniture_boxes', [])
        
        try:
            controller.reset(scene=house_data)
        except Exception as e:
            print(f"Failed to load {scene_name}: {e}")
            continue

        # Find a clear spawn point
        spawn_x, spawn_z = l / 2.0, w / 2.0
        found = False
        for x in np.arange(0.25, l, 0.25):
            for z in np.arange(0.25, w, 0.25):
                if is_clear(x, z, furniture_boxes, l, w):
                    spawn_x, spawn_z = x, z
                    found = True
                    break
            if found: break
            
        controller.step(action="Teleport", position={"x": spawn_x, "y": 0.9, "z": spawn_z}, forceAction=True)
        
        rp_event = controller.step(action="GetReachablePositions")
        if not rp_event.metadata["lastActionSuccess"]:
            continue
            
        reachable_positions = rp_event.metadata["actionReturn"]
        if not reachable_positions:
            continue
            
        scene_objects = 0
        scene_accessible = 0
        
        # Test accessibility to every object
        for idx, box in enumerate(furniture_boxes):
            min_x, max_x, min_z, max_z = box
            
            # An object is accessible if there is a reachable NavMesh point within 0.75m of its edge
            is_accessible = False
            for p in reachable_positions:
                px, pz = p['x'], p['z']
                # Distance to the bounding box
                dx = max(min_x - px, 0, px - max_x)
                dz = max(min_z - pz, 0, pz - max_z)
                dist_to_box = math.hypot(dx, dz)
                
                if dist_to_box <= 0.75:
                    is_accessible = True
                    break
                    
            scene_objects += 1
            if is_accessible:
                scene_accessible += 1
                
        if scene_objects > 0:
            results[scene_name] = {
                "total_objects": scene_objects,
                "accessible_objects": scene_accessible,
                "accessibility_rate": scene_accessible / scene_objects
            }
            total_objects_evaluated += scene_objects
            total_objects_accessible += scene_accessible
            
    controller.stop()
    
    avg_accessibility = total_objects_accessible / total_objects_evaluated if total_objects_evaluated > 0 else 0
    
    # Save results
    save_path = os.path.join(scenes_dir, "navigation_results.json")
    with open(save_path, 'w') as f:
        json.dump({
            "summary": {
                "instance_count": len(results),
                "total_objects_evaluated": total_objects_evaluated,
                "total_objects_accessible": total_objects_accessible,
                "average_accessibility_rate": avg_accessibility
            },
            "scenes": results
        }, f, indent=2)
        
    print(f"\nNavigation Evaluation Complete for {scenes_dir}!")
    print(f"Overall Accessibility Rate: {avg_accessibility * 100:.2f}% ({total_objects_accessible}/{total_objects_evaluated} objects)")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Navigation/Accessibility for ProcTHOR JSON scenes.")
    parser.add_argument("--scenes_dir", type=str, required=True, help="Directory containing the converted JSON files")
    args = parser.parse_args()
    
    evaluate_navigation(args.scenes_dir)
