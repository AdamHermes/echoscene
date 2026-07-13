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

def parse_final_json(file_path, full=False):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    class_map = {
        1: 'bed', 2: 'bookshelf', 3: 'cabinet', 4: 'chair', 5: 'desk', 
        6: 'floor', 7: 'lamp', 8: 'nightstand', 9: 'shelf', 10: 'sofa', 
        11: 'table', 12: 'tv_stand', 13: 'wardrobe'
    }
    
    scenes = {}
    
    scene_ids_to_process = data['scene_ids']
    if not full:
        scene_ids_to_process = scene_ids_to_process[:50]
        
    for i, scene_id in enumerate(scene_ids_to_process):
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
        
        l = max_x - min_x
        w = max_z - min_z
        fx = (max_x + min_x) / 2.0
        fz = (max_z + min_z) / 2.0
    else:
        l = scene_data['floor']['size']['x']
        w = scene_data['floor']['size']['z']
        fx = scene_data['floor']['position']['x']
        fz = scene_data['floor']['position']['z']
    
    half_l = l / 2.0
    half_w = w / 2.0
    
    # Calculate offset to shift everything to positive quadrant
    offset_x = -(fx - half_l)
    offset_z = -(fz - half_w)
    
    PAD = 0.5  # Expand floor polygon beyond walls so agent capsule never exceeds Unity sceneBounds

    # Floor polygon: 0.5m larger on all sides than the actual walls.
    # Unity uses floorPolygon to calculate sceneBounds. The agent capsule sweeps ~0.3m past
    # the inner wall face, which would exceed bounds if the polygon == room. This padding prevents crash.
    fp1 = {"x": -PAD,   "y": 0, "z": -PAD}
    fp2 = {"x": -PAD,   "y": 0, "z": w+PAD}
    fp3 = {"x": l+PAD,  "y": 0, "z": w+PAD}
    fp4 = {"x": l+PAD,  "y": 0, "z": -PAD}
    polygon = [fp1, fp2, fp3, fp4]

    # Actual room walls stay at exact room dimensions (l x w)
    p1 = {"x": 0.0, "y": 0, "z": 0.0}
    p2 = {"x": 0.0, "y": 0, "z": w}
    p3 = {"x": l,   "y": 0, "z": w}
    p4 = {"x": l,   "y": 0, "z": 0.0}

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
                {"x": vb['x'], "y": 3, "z": vb['z']},
                {"x": va['x'], "y": 3, "z": va['z']}
            ]
        })
        
    procthor_objects = []
    furniture_boxes = []   # list of (min_x, max_x, min_z, max_z) for each piece of furniture

    import math as _math
    for idx, obj in enumerate(scene_data['objects']):
        cx = obj['position']['x'] + offset_x
        cz = obj['position']['z'] + offset_z
        obj_w = obj['size']['x']
        obj_d = obj['size']['z']
        
        # Calculate actual min_y and max_y of the object
        min_y = obj['position']['y'] - obj['size']['y'] / 2.0
        max_y = obj['position']['y'] + obj['size']['y'] / 2.0
        
        # Ensure it doesn't sink below floor (sometimes bounds are slightly off)
        min_y = max(0.0, min_y)
        
        theta = obj['rotation']['y']

        # Build the 4 corners of the furniture footprint (axis-aligned for now — rotation handled below)
        hw, hd = obj_w / 2.0, obj_d / 2.0
        # theta from final.json is already in radians!
        rad = theta
        cos_t, sin_t = _math.cos(rad), _math.sin(rad)

        def rot(lx, lz):
            return cx + lx*cos_t - lz*sin_t, cz + lx*sin_t + lz*cos_t

        # 4 corners in order
        c0 = rot(-hw, -hd)
        c1 = rot( hw, -hd)
        c2 = rot( hw,  hd)
        c3 = rot(-hw,  hd)

        corners = [c0, c1, c2, c3]
        
        # Track axis-aligned bounding box for spawn-point selection
        # ONLY if the object is low enough to block the agent (e.g. min_y < 1.0)
        if min_y < 1.0:
            xs = [p[0] for p in corners]; zs = [p[1] for p in corners]
            furniture_boxes.append((min(xs), max(xs), min(zs), max(zs)))

        # Must be CLOCKWISE so normals face outward in Unity's left-handed system!
        edges = [(c0, c3), (c3, c2), (c2, c1), (c1, c0)]
        for e_idx, (va, vb) in enumerate(edges):
            wall_id = f"furn|{idx}|{e_idx}"
            walls.append({
                "id": wall_id,
                "roomId": "room|0",
                "material": {"name": "PureWhite"},
                "polygon": [
                    {"x": va[0], "y": min_y, "z": va[1]},
                    {"x": vb[0], "y": min_y, "z": vb[1]},
                    {"x": vb[0], "y": 3.0, "z": vb[1]},
                    {"x": va[0], "y": 3.0, "z": va[1]},
                ]
            })
            
        # Add a horizontal lid (ceiling) to the furniture box to block NavMesh voxels from falling inside
        # Must be CLOCKWISE from top-down perspective to face upwards!
        lid_corners = [c0, c1, c2, c3]
        walls.append({
            "id": f"furn|{idx}|lid",
            "roomId": "room|0",
            "material": {"name": "PureWhite"},
            "polygon": [
                {"x": lid_corners[0][0], "y": 3.0, "z": lid_corners[0][1]},
                {"x": lid_corners[1][0], "y": 3.0, "z": lid_corners[1][1]},
                {"x": lid_corners[2][0], "y": 3.0, "z": lid_corners[2][1]},
                {"x": lid_corners[3][0], "y": 3.0, "z": lid_corners[3][1]}
            ]
        })

        
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
        "ceilings": [],
        "floorMaterial": {"name": "WoodFloorsCross"},
        "floorPolygon": polygon
    }]
    house["walls"] = walls
    house["objects"] = procthor_objects
    house["doors"] = []
    house["windows"] = []
    # Store furniture bounding boxes so eval_walkability can pick a clear spawn point
    house["furniture_boxes"] = furniture_boxes
    house["room_dims"] = {"l": l, "w": w}

    # Spawn agent at near-corner (0.25, 0.25) — furniture is clamped away from walls
    # so this corner is always guaranteed to be outside all furniture boxes.
    agent_x = 0.25
    agent_z = 0.25

    # Full ProcTHOR-compatible metadata format (matches procthor-10k houses exactly)
    agent_pose = {
        "horizon": 30,
        "position": {"x": agent_x, "y": 0.95, "z": agent_z},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "standing": True
    }
    house["metadata"] = {
        "schema": "1.0.0",
        "agent": agent_pose,
        "agentPoses": {
            "default": agent_pose,
            "arm": agent_pose,
            "locobot": {**agent_pose, "standing": None},
            "stretch": agent_pose
        }
    }
    house["proceduralParameters"] = {
        "floorColliderThickness": 1.0,
        "receptacleHeight": 0.7,
        "skyboxId": "Sky1",
        "ceilingMaterial": {"name": "PureWhite"},
        "lights": [
            {
                "id": "DirectionalLight", "type": "directional",
                "intensity": 1.0, "indirectMultiplier": 1.0,
                "position": {"x": 0.84, "y": 0.1855, "z": -1.09},
                "rotation": {"x": 66, "y": 75, "z": 0},
                "rgb": {"r": 1.0, "g": 1.0, "b": 1.0},
                "shadow": {"type": "Soft", "strength": 1, "normalBias": 0,
                           "bias": 0, "nearPlane": 0.2, "resolution": "FromQualitySettings"}
            }
        ]
    }

    return house

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert echoscene bounding boxes to ProcTHOR JSON scenes.")
    parser.add_argument("--bbox_path", type=str, required=True, help="Path to the debug_bbox.txt or final.json file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the converted JSON files")
    parser.add_argument("--full", action="store_true", help="If specified, converts all scenes. If not, limits to the first 50 scenes.")
    args = parser.parse_args()
    
    debug_bbox_path = args.bbox_path
    out_dir = args.out_dir
    
    if debug_bbox_path.endswith('.json'):
        scenes = parse_final_json(debug_bbox_path, full=args.full)
    else:
        scenes = parse_debug_bbox(debug_bbox_path)
        
    print(f"Parsed {len(scenes)} scenes from {debug_bbox_path}.")
    
    os.makedirs(out_dir, exist_ok=True)
    
    for scene_name, scene_data in scenes.items():
        house_json = convert_to_procthor_json(scene_name, scene_data)
        if house_json:
            house_json["objects_raw"] = scene_data.get('objects', [])
            out_path = os.path.join(out_dir, f"{scene_name}.json")
            with open(out_path, 'w') as f:
                json.dump(house_json, f, indent=2)
            
    print(f"Converted scenes saved to {out_dir}")
