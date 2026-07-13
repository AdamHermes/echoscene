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

    # Real room dimensions stored at conversion time (excludes the 0.5m sceneBounds padding)
    rd = house_data.get('room_dims', {})
    l = rd.get('l', 0)
    w = rd.get('w', 0)
    true_total_area = l * w

    # Real room occupies [0, l] x [0, w] in scene space
    room_min_x, room_max_x = 0.0, l
    room_min_z, room_max_z = 0.0, w

    # 1. Load scene
    controller.reset(scene=house_data)

    # 2. Find a clear teleport position
    furniture_boxes = house_data.get('furniture_boxes', [])
    def is_clear(x, z):
        clearance = 0.3
        for box in furniture_boxes:
            min_x, max_x, min_z, max_z = box
            if (min_x - clearance <= x <= max_x + clearance) and (min_z - clearance <= z <= max_z + clearance):
                return False
        if x < clearance or x > l - clearance or z < clearance or z > w - clearance:
            return False
        return True

    spawn_x, spawn_z = l / 2.0, w / 2.0  # fallback
    found = False
    x = 0.25
    while x < l and not found:
        z = 0.25
        while z < w:
            if is_clear(x, z):
                spawn_x, spawn_z = x, z
                found = True
                break
            z += 0.25
        x += 0.25

    event_tp = controller.step(action="Teleport", position={"x": spawn_x, "y": 0.9, "z": spawn_z}, forceAction=True)
    print(f"Teleport success to ({spawn_x:.2f}, {spawn_z:.2f}): {event_tp.metadata['lastActionSuccess']}")

    # 3. Get NavMesh reachable positions
    event = controller.step(action="GetReachablePositions")

    if event.metadata["lastActionSuccess"]:
        reachable_positions = event.metadata["actionReturn"]
        # Filter: only count points inside the real room walls (exclude 0.5m padding zone)
        valid_positions = [p for p in reachable_positions
                          if room_min_x <= p['x'] <= room_max_x and room_min_z <= p['z'] <= room_max_z]

        grid_size = 0.25
        walkable_area = len(valid_positions) * (grid_size ** 2)
        walkability_score = walkable_area / true_total_area if true_total_area > 0 else 0

        return walkability_score, walkable_area, true_total_area, len(valid_positions)
    else:
        print(f"Failed to get reachable positions: {event.metadata['errorMessage']}")
        return 0, 0, true_total_area, 0


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
