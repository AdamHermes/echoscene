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