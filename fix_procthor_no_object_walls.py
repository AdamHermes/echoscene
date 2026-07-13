import re

with open("convert_echoscene_to_procthor.py", "r") as f:
    content = f.read()

new_logic = """
    # NO object walls at all. Only procthor_objects.
    procthor_objects = []
    shifted_obj_polys = []
    
    for idx, obj in enumerate(scene_data['objects']):
        cx = obj['position']['x'] + offset_x
        cz = obj['position']['z'] + offset_z
        sx = obj['size']['x']
        sz = obj['size']['z']
        
        theta = math.radians(obj['rotation']['y'])
        hw = sx / 2.0
        hd = sz / 2.0
        local_corners = [(hw, hd), (hw, -hd), (-hw, -hd), (-hw, hd)]
        world_corners = []
        for lx, lz in local_corners:
            rx = lx * math.cos(theta) - lz * math.sin(theta)
            rz = lx * math.sin(theta) + lz * math.cos(theta)
            world_corners.append((cx + rx, cz + rz))
        
        shifted_obj_polys.append(shapely.geometry.Polygon(world_corners))
        
        # WE OMIT OBJECT WALLS ENTIRELY TO PREVENT BUILDER CRASHES

    # Grid search for a valid agent point
    import numpy as np
    margin = 0.5
    best_spawn = (fx + offset_x, fz + offset_z)
    found = False
    for x_cand in np.arange(margin, l - margin, 0.25):
        for z_cand in np.arange(margin, w - margin, 0.25):
            pt = shapely.geometry.Point(x_cand, z_cand)
            valid = True
            for poly in shifted_obj_polys:
                if poly.distance(pt) < 0.4:
                    valid = False
                    break
            if valid:
                best_spawn = (float(x_cand), float(z_cand))
                found = True
                break
        if found:
            break

    try:
        with open("sample_house.json", "r") as f:
            house = json.load(f)
    except FileNotFoundError:
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
    house["objects"] = [] # WE OMIT PROCTHOR_OBJECTS TOO to avoid asset ID crashes!
    house["doors"] = []
    house["windows"] = []
    
    if "metadata" not in house:
        house["metadata"] = {}
    if "agent" not in house["metadata"]:
        house["metadata"]["agent"] = {}
    house["metadata"]["agent"]["spawn_x"] = best_spawn[0]
    house["metadata"]["agent"]["spawn_z"] = best_spawn[1]
    house["metadata"]["agent"]["position"] = {"x": best_spawn[0], "y": 0.9, "z": best_spawn[1]}
    house["metadata"]["agent"]["rotation"] = {"x": 0, "y": 0, "z": 0}
    
    return house
"""

pattern = r"    # Convert objects to .*?return house"
content = re.sub(pattern, new_logic.strip("\n"), content, flags=re.DOTALL)

with open("convert_echoscene_to_procthor.py", "w") as f:
    f.write(content)

