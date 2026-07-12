import os
import json
import math

# Mapping from EchoScene class names to AI2-THOR/ProcTHOR asset types
CLASS_MAPPING = {
    'table': 'DiningTable',
    'nightstand': 'SideTable',
    'chair': 'Chair',
    'bed': 'Bed',
    'cabinet': 'Cabinet',
    'lamp': 'FloorLamp',
    'wardrobe': 'Dresser',
    'tv_stand': 'TVStand',
    'desk': 'Desk',
    'sofa': 'Sofa',
    'bookshelf': 'Bookcase',
    # Add more mappings as encountered
}

def parse_debug_bbox(file_path):
    scenes = {}
    current_scene = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("SCENE:"):
                # e.g., "SCENE: SecondBedroom-860  |  7 objects"
                scene_name = line.split("|")[0].replace("SCENE:", "").strip()
                current_scene = scene_name
                scenes[current_scene] = {'objects': [], 'floor': None}
            elif current_scene and not line.startswith("=") and not line.startswith("-") and not line.startswith("Obj"):
                # Parse object line
                # e.g., "chair                 0.655  1.026  0.722   -0.540 -0.001 -0.925     0.00°"
                parts = line.split()
                if len(parts) >= 8:
                    name = parts[0]
                    l, h, w = map(float, parts[1:4])
                    x, y, z = map(float, parts[4:7])
                    angle = float(parts[7].replace('°', ''))
                    
                    obj_data = {
                        'class': name,
                        'size': {'x': l, 'y': h, 'z': w},
                        'position': {'x': x, 'y': y, 'z': z},
                        'rotation': {'x': 0, 'y': angle, 'z': 0}
                    }
                    
                    if name == 'floor':
                        scenes[current_scene]['floor'] = obj_data
                    elif name != '_scene_':
                        scenes[current_scene]['objects'].append(obj_data)
    return scenes

def parse_final_json(file_path):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    class_map = {
        1: 'bed', 2: 'bookshelf', 3: 'cabinet', 4: 'chair', 5: 'desk', 
        6: 'floor', 7: 'lamp', 8: 'nightstand', 9: 'shelf', 10: 'sofa', 
        11: 'table', 12: 'tv_stand', 13: 'wardrobe'
    }
    
    scenes = {}
    for i, scene_id in enumerate(data['scene_ids']):
        scene_objs = []
        floor = None
        labels = data['class_labels'][i]
        sizes = data['sizes'][i]
        translations = data['translations'][i]
        angles = data['angles'][i]
        
        for j in range(len(labels)):
            try:
                class_idx = labels[j].index(1.0)
            except ValueError:
                continue
                
            if class_idx == 14: # padding
                continue
            
            name = class_map.get(class_idx, f"unknown_{class_idx}")
            l, h, w = sizes[j]
            x, y, z = translations[j]
            angle = angles[j][0]
            
            obj_data = {
                'class': name,
                'size': {'x': l, 'y': h, 'z': w},
                'position': {'x': x, 'y': y, 'z': z},
                'rotation': {'x': 0, 'y': angle, 'z': 0}
            }

            if name == 'floor':
                floor = obj_data
            else:
                scene_objs.append(obj_data)
        scenes[scene_id] = {'objects': scene_objs, 'floor': floor}
    return scenes

def convert_to_procthor_json(scene_name, scene_data):
    if not scene_data['floor']:
        # Synthesize a floor based on the extent of all objects
        objs = scene_data.get('objects', [])
        if not objs:
            return None
        min_x = min(o['position']['x'] - o['size']['x']/2 for o in objs)
        max_x = max(o['position']['x'] + o['size']['x']/2 for o in objs)
        min_z = min(o['position']['z'] - o['size']['z']/2 for o in objs)
        max_z = max(o['position']['z'] + o['size']['z']/2 for o in objs)
        
        # Add a 0.5m padding to the walls
        min_x -= 0.5
        max_x += 0.5
        min_z -= 0.5
        max_z += 0.5
        
        l = max_x - min_x
        w = max_z - min_z
        fx = (max_x + min_x) / 2.0
        fz = (max_z + min_z) / 2.0
    else:
        fx = scene_data['floor']['position']['x']
        fz = scene_data['floor']['position']['z']
        l = scene_data['floor']['size']['x']
        w = scene_data['floor']['size']['z']
    
    half_l = l / 2.0
    half_w = w / 2.0
    
    # Calculate offset to shift everything to positive quadrant
    offset_x = -(fx - half_l)
    offset_z = -(fz - half_w)
    
    # Floor polygon (shifted)
    p1 = {"x": 0.0, "y": 0, "z": 0.0}
    p2 = {"x": 0.0, "y": 0, "z": w}
    p3 = {"x": l, "y": 0, "z": w}
    p4 = {"x": l, "y": 0, "z": 0.0}
    polygon = [p1, p2, p3, p4]

    # Generate walls for the room to contain the agent (shifted)
    walls = []
    edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
    for i, (va, vb) in enumerate(edges):
        wall_id = f"wall|0|{va['x']:.2f}|{va['z']:.2f}|{vb['x']:.2f}|{vb['z']:.2f}"
        walls.append({
            "id": wall_id,
            "roomId": "room|0",
            "color": {"r": 1.0, "g": 1.0, "b": 1.0},
            "material": {
                "name": "PureWhite",
                "color": {"r": 1.0, "g": 1.0, "b": 1.0}
            },
            "polygon": [
                {"x": va['x'], "y": 0, "z": va['z']},
                {"x": vb['x'], "y": 0, "z": vb['z']},
                {"x": va['x'], "y": 3, "z": va['z']},
                {"x": vb['x'], "y": 3, "z": vb['z']}
            ]
        })
    
    # Generate object barriers as walls for accurate Walkability bounding boxes
    obj_wall_idx = len(edges)
    procthor_objects = []
    
    for idx, obj in enumerate(scene_data['objects']):
        # We still want to map classes if we wanted to spawn them, 
        # but for walkability, we use exact bounding box walls!
        
        cx = obj['position']['x'] + offset_x
        cz = obj['position']['z'] + offset_z
        w = obj['size']['x']
        d = obj['size']['z']
        
        # Convert rotation to radians
        # Note: AI2-THOR is left-handed, standard math is right-handed. 
        # But for bounding box perimeter, the direction doesn't matter as much.
        theta = math.radians(obj['rotation']['y'])
        
        hw = w / 2
        hd = d / 2
        
        local_corners = [
            (hw, hd),
            (hw, -hd),
            (-hw, -hd),
            (-hw, hd)
        ]
        
        world_corners = []
        for lx, lz in local_corners:
            rx = lx * math.cos(theta) - lz * math.sin(theta)
            rz = lx * math.sin(theta) + lz * math.cos(theta)
            world_corners.append({'x': cx + rx, 'z': cz + rz})
            
        obj_edges = [
            (world_corners[0], world_corners[1]),
            (world_corners[1], world_corners[2]),
            (world_corners[2], world_corners[3]),
            (world_corners[3], world_corners[0])
        ]
        
        # We give the object walls a distinct red color for visualization
        for j, (va, vb) in enumerate(obj_edges):
            # wall_id = f"wall|0|{va['x']:.2f}|{va['z']:.2f}|{vb['x']:.2f}|{vb['z']:.2f}_{obj_wall_idx}"
            # Ensure wall ID doesn't strictly break if we append an index, but to be safe, use standard coords
            wall_id = f"wall|0|{va['x']:.3f}|{va['z']:.3f}|{vb['x']:.3f}|{vb['z']:.3f}"
            walls.append({
                "id": wall_id,
                "roomId": "room|0",
                "color": {"r": 1.0, "g": 0.0, "b": 0.0},
                "material": {
                    "name": "PureWhite",
                    "color": {"r": 1.0, "g": 0.0, "b": 0.0}
                },
                "polygon": [
                    {"x": va['x'], "y": 0, "z": va['z']},
                    {"x": vb['x'], "y": 0, "z": vb['z']},
                    {"x": va['x'], "y": obj['size']['y'], "z": va['z']},
                    {"x": vb['x'], "y": obj['size']['y'], "z": vb['z']}
                ]
            })
            obj_wall_idx += 1
        
    try:
        with open("sample_house.json", "r") as f:
            house = json.load(f)
    except FileNotFoundError:
        # Fallback if not found (though it should be there since we just generated it)
        house = {"metadata": {"schema": "1.0.0"}, "proceduralParameters": {}}
        
    house["rooms"] = [{
        "id": "room|0",
        "roomType": "Bedroom",
        "children": [],
        "floorMaterial": {"name": "WoodFloorsCross"},
        "floorPolygon": polygon,
        "ceilings": []
    }]
    house["walls"] = walls
    house["objects"] = procthor_objects
    house["doors"] = []
    house["windows"] = []
    
    # Update agent spawn
    if "metadata" not in house:
        house["metadata"] = {}
    if "agent" not in house["metadata"]:
        house["metadata"]["agent"] = {}
    house["metadata"]["agent"]["spawn_x"] = fx + offset_x
    house["metadata"]["agent"]["spawn_z"] = fz + offset_z
    house["metadata"]["agent"]["position"] = {"x": fx + offset_x, "y": 0.9, "z": fz + offset_z}
    house["metadata"]["agent"]["rotation"] = {"x": 0, "y": 0, "z": 0}
    
    return house

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert echoscene bounding boxes to ProcTHOR JSON scenes.")
    parser.add_argument("--bbox_path", type=str, required=True, help="Path to the debug_bbox.txt file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the converted JSON files")
    args = parser.parse_args()
    
    debug_bbox_path = args.bbox_path
    out_dir = args.out_dir
    
    if debug_bbox_path.endswith('.json'):
        scenes = parse_final_json(debug_bbox_path)
    else:
        scenes = parse_debug_bbox(debug_bbox_path)
        
    print(f"Parsed {len(scenes)} scenes from {debug_bbox_path}.")
    
    os.makedirs(out_dir, exist_ok=True)
    
    for scene_name, scene_data in scenes.items():
        house_json = convert_to_procthor_json(scene_name, scene_data)
        if house_json:
            out_path = os.path.join(out_dir, f"{scene_name}.json")
            with open(out_path, 'w') as f:
                json.dump(house_json, f, indent=2)
            
    print(f"Converted scenes saved to {out_dir}")
