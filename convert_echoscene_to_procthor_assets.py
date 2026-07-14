import os
import json
import math

CLASS_MAPPING = {
    'table': 'Coffee_Table_227_1_1',
    'nightstand': 'Side_Table_307_1',
    'chair': 'Chair_223_1',
    'bed': 'Bed_13',
    'cabinet': 'Dresser_413_1',
    'lamp': 'Floor_Lamp_18',
    'wardrobe': 'Dresser_413_1',
    'tv_stand': 'TV_Stand_206_3',
    'desk': 'Desk_323_1',
    'sofa': 'Sofa_221_1',
    'bookshelf': 'TV_Stand_206_3',
}

def convert_to_procthor_house_assets(final_json_path, out_dir):
    with open(final_json_path, 'r') as f:
        data = json.load(f)
        
    class_map = {
        1: 'bed', 2: 'bookshelf', 3: 'cabinet', 4: 'chair', 5: 'desk', 
        6: 'floor', 7: 'lamp', 8: 'nightstand', 9: 'shelf', 10: 'sofa', 
        11: 'table', 12: 'tv_stand', 13: 'wardrobe'
    }
    
    os.makedirs(out_dir, exist_ok=True)
    
    scene_ids_to_process = data['scene_ids'][:50] # Just do first 50 for speed
    for i, scene_id in enumerate(scene_ids_to_process):
        labels = data['class_labels'][i]
        sizes = data['sizes'][i]
        translations = data['translations'][i]
        angles = data['angles'][i]
        
        objects = []
        floor_obj = None
        for j in range(len(labels)):
            try:
                class_idx = labels[j].index(1.0)
            except ValueError:
                continue
            if class_idx == 14:
                continue
                
            name = class_map.get(class_idx, f"unknown_{class_idx}")
            l, h, w = sizes[j]
            x, y, z = translations[j]
            angle = angles[j][0]
            
            if name == 'floor':
                floor_obj = {'size': {'x': l, 'y': h, 'z': w}, 'position': {'x': x, 'y': y, 'z': z}}
            else:
                asset_id = CLASS_MAPPING.get(name, 'Chair_1') # Default fallback
                obj = {
                    'assetId': asset_id,
                    'id': f"{name}_{j}",
                    'kinematic': True,
                    'position': {'x': x, 'y': y, 'z': z},
                    # Convert radians back to degrees and invert if necessary
                    'rotation': {'x': 0, 'y': angle * 180 / math.pi, 'z': 0},
                    # We pass boundingBox for downstream physics logic, but without exact scaling logic,
                    # the visual asset might not match this box size in Unity.
                    'boundingBox': {'x': l, 'y': h, 'z': w} 
                }
                objects.append(obj)
                
        if not floor_obj:
            if not objects:
                continue
            min_x = min(o['position']['x'] - o['boundingBox']['x']/2 for o in objects)
            max_x = max(o['position']['x'] + o['boundingBox']['x']/2 for o in objects)
            min_z = min(o['position']['z'] - o['boundingBox']['z']/2 for o in objects)
            max_z = max(o['position']['z'] + o['boundingBox']['z']/2 for o in objects)
            floor_obj = {
                'size': {'x': max_x - min_x, 'y': 0.1, 'z': max_z - min_z},
                'position': {'x': (max_x + min_x) / 2.0, 'y': 0, 'z': (max_z + min_z) / 2.0}
            }
            
        offset_x = (floor_obj['size']['x'] / 2.0) - floor_obj['position']['x']
        offset_z = (floor_obj['size']['z'] / 2.0) - floor_obj['position']['z']
        
        for obj in objects:
            obj['position']['x'] += offset_x
            obj['position']['z'] += offset_z
            
        l, w = floor_obj['size']['x'], floor_obj['size']['z']
        
        # Remove boundingBox so ProcTHOR doesn't distort the meshes
        for obj in objects:
            if 'boundingBox' in obj:
                del obj['boundingBox']
        
        floor_polygon = [
            {'x': 0.0, 'z': 0.0},
            {'x': l, 'z': 0.0},
            {'x': l, 'z': w},
            {'x': 0.0, 'z': w}
        ]
        
        agent_pose = {
            "horizon": 30,
            "position": {"x": 1.0, "y": 0.95, "z": 1.0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "standing": True
        }
        procthor_house = {
            "metadata": {
                "schema": "1.0.0",
                "agent": agent_pose,
                "agentPoses": {
                    "default": agent_pose,
                    "arm": agent_pose,
                    "locobot": {**agent_pose, "standing": None},
                    "stretch": agent_pose
                }
            },
            "rooms": [
                {
                    "id": "room|0",
                    "roomType": "Bedroom",
                    "children": [],
                    "ceilings": [],
                    "floorMaterial": {"name": "WoodFloorsCross"},
                    "floorPolygon": floor_polygon
                }
            ],
            "walls": [],
            "objects": objects,
            "doors": [],
            "windows": [],
            "proceduralParameters": {
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
        }
        
        out_path = os.path.join(out_dir, f"{scene_id}.json")
        with open(out_path, 'w') as f:
            json.dump(procthor_house, f, indent=2)

if __name__ == "__main__":
    convert_to_procthor_house_assets("./output/baseline/vis/2050/final.json", "./output/baseline_assets_scenes")
    print("Successfully converted baseline to true asset scenes!")
