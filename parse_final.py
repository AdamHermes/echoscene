import json

def parse_final_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    class_map = {
        1: 'bed', 2: 'bookshelf', 3: 'cabinet', 4: 'chair', 5: 'desk', 
        6: 'floor', 7: 'lamp', 8: 'nightstand', 9: 'shelf', 10: 'sofa', 
        11: 'table', 12: 'tv_stand', 13: 'wardrobe'
    }
    
    scenes = {}
    for i, scene_id in enumerate(data['scene_ids']):
        scene_objs = []
        labels = data['class_labels'][i]
        sizes = data['sizes'][i]
        translations = data['translations'][i]
        angles = data['angles'][i]
        
        for j in range(len(labels)):
            # Find the index of 1.0
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
            
            scene_objs.append({
                'name': name,
                'l': l, 'h': h, 'w': w,
                'x': x, 'y': y, 'z': z,
                'angle': angle
            })
        scenes[scene_id] = scene_objs
    return scenes

