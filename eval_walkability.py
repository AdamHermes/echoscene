import os
import json
import glob
import argparse

try:
    from ai2thor.controller import Controller
except ImportError:
    print("Warning: ai2thor is not installed. Please install it using 'pip install ai2thor prior' to running this script.")
    exit(1)

def calculate_walkability(scene_json_path, controller):
    with open(scene_json_path, 'r') as f:
        house_data = json.load(f)
    
    # Extract the floor polygon to compute total expected area
    floor_poly = house_data['rooms'][0]['floorPolygon']
    x_coords = [p['x'] for p in floor_poly]
    z_coords = [p['z'] for p in floor_poly]
    
    length = max(x_coords) - min(x_coords)
    width = max(z_coords) - min(z_coords)
    total_area = length * width
    
    # 1. Initialize environment with the generated scene
    controller.reset(scene=house_data)
    
    # 1.5 Explicitly teleport agent to the center of the floor to prevent out-of-bounds error
    fx = (max(x_coords) + min(x_coords)) / 2.0
    fz = (max(z_coords) + min(z_coords)) / 2.0
    event_tp = controller.step(action="Teleport", position={"x": fx, "y": 0.9, "z": fz}, forceAction=True)
    print(f"Teleport success: {event_tp.metadata['lastActionSuccess']}")
    
    # 2. Get reachable positions via NavMesh
    event = controller.step(action="GetReachablePositions")
    
    if event.metadata["lastActionSuccess"]:
        reachable_positions = event.metadata["actionReturn"]
        
        # Grid size in ProcTHOR is typically 0.25m
        grid_size = 0.25
        walkable_area = len(reachable_positions) * (grid_size ** 2)
        
        walkability_score = walkable_area / total_area if total_area > 0 else 0
        
        return walkability_score, walkable_area, total_area, len(reachable_positions)
    else:
        print(f"Failed to get reachable positions: {event.metadata['errorMessage']}")
        return 0, 0, total_area, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Walkability for ProcTHOR JSON scenes.")
    parser.add_argument("--scenes_dir", type=str, required=True, help="Directory containing the converted JSON files")
    args = parser.parse_args()
    
    scenes_dir = args.scenes_dir
    scene_files = glob.glob(os.path.join(scenes_dir, "*.json"))
    
    if not scene_files:
        print(f"No converted scenes found in {scenes_dir}.")
        print("Please run convert_echoscene_to_procthor.py first.")
        exit(1)
        
    print(f"Found {len(scene_files)} scenes to evaluate.")
    
    # Start ProcTHOR Controller once
    print("Initializing AI2-THOR Controller...")
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="Procedural", 
        gridSize=0.25
    )
    
    results = {}
    
    for scene_file in scene_files:
        if scene_file.endswith("walkability_results.json"):
            continue
        scene_name = os.path.basename(scene_file).replace('.json', '')
        print(f"\nEvaluating Walkability for: {scene_name}")
        
        score, walkable, total, points = calculate_walkability(scene_file, controller)
        
        results[scene_name] = {
            'walkability_score': score,
            'walkable_area': walkable,
            'total_area': total,
            'navigable_points': points
        }
        
        print(f"Score: {score:.2%} ({walkable:.2f}m² / {total:.2f}m², Points: {points})")
        
    controller.stop()
    
    # Save results
    avg_score = sum([res['walkability_score'] for res in results.values()]) / len(results) if results else 0
    final_output = {
        "summary": {
            "instance_count": len(results),
            "average_walkability": avg_score
        },
        "scenes": results
    }
    
    results_path = os.path.join(scenes_dir, "walkability_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nEvaluation complete! Results saved to {results_path}")
