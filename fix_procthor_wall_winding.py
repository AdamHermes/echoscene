import re

with open("convert_echoscene_to_procthor.py", "r") as f:
    content = f.read()

# I need to fix the wall winding order for BOTH the outer walls AND any object walls.
# Let's completely rewrite the conversion logic from scratch to be clean.

new_logic = """
    # Calculate offset to shift everything to positive quadrant
    offset_x = -(fx - half_l)
    offset_z = -(fz - half_w)
    
    # Floor polygon (shifted)
    p1 = {"x": 0.0, "y": 0, "z": 0.0}
    p2 = {"x": 0.0, "y": 0, "z": w}
    p3 = {"x": l, "y": 0, "z": w}
    p4 = {"x": l, "y": 0, "z": 0.0}
    # CW for floor to face up
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
            # CW winding so normal points INWARDS
            "polygon": [
                {"x": va['x'], "y": 0, "z": va['z']},
                {"x": va['x'], "y": 3, "z": va['z']},
                {"x": vb['x'], "y": 3, "z": vb['z']},
                {"x": vb['x'], "y": 0, "z": vb['z']}
            ]
        })
        
    import shapely.geometry
    import shapely.ops
    import numpy as np
    
    # We will use unary_union to merge all objects, and generate inward-facing walls for them
    # BUT we will use a separate roomId to NOT break room|0
    
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
            
            # clamp to room bounds to avoid walls leaking outside
            wx = max(0.01, min(cx + rx, l - 0.01))
            wz = max(0.01, min(cz + rz, w - 0.01))
            world_corners.append((wx, wz))
            
        shifted_obj_polys.append(shapely.geometry.Polygon(world_corners))
        
    unioned = shapely.ops.unary_union(shifted_obj_polys)
    
    if isinstance(unioned, shapely.geometry.Polygon):
        merged_polys = [unioned]
    elif isinstance(unioned, shapely.geometry.MultiPolygon):
        merged_polys = list(unioned.geoms)
    else:
        merged_polys = []
        
    for p_idx, poly in enumerate(merged_polys):
        rings = [poly.exterior] + list(poly.interiors)
        for r_idx, ring in enumerate(rings):
            coords = list(ring.coords)
            # Reverse the coordinates so the normals face OUTWARDS from the object (into the room)
            # shapely exterior rings are CCW, so if we reverse them they become CW
            # Wait, if they are CW, and we draw walls va->vb, the normal points right.
            # We want the normal to point AWAY from the object.
            # Let's just generate the walls and ensure CW winding for the wall face itself!
            for i in range(len(coords) - 1):
                va = coords[i]
                vb = coords[i+1]
                wall_id = f"wall|1|{va[0]:.3f}|{va[1]:.3f}|{vb[0]:.3f}|{vb[1]:.3f}_{p_idx}_{r_idx}"
                walls.append({
                    "id": wall_id,
                    "roomId": "room|1",
                    "color": {"r": 1.0, "g": 0.0, "b": 0.0},
                    "material": {
                        "name": "PureWhite",
                        "color": {"r": 1.0, "g": 0.0, "b": 0.0}
                    },
                    "polygon": [
                        {"x": va[0], "y": 0, "z": va[1]},
                        {"x": va[0], "y": 2.0, "z": va[1]},
                        {"x": vb[0], "y": 2.0, "z": vb[1]},
                        {"x": vb[0], "y": 0, "z": vb[1]}
                    ]
                })

    # Grid search for a valid agent point
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
    }, {
        "id": "room|1",
        "roomType": "Bedroom",
        "children": [],
        "floorMaterial": {"name": "WoodFloorsCross"},
        "floorPolygon": polygon,
        "ceilings": []
    }]
    house["walls"] = walls
    house["objects"] = []
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

pattern = r"    # Calculate offset to shift everything to positive quadrant.*?return house"
content = re.sub(pattern, new_logic.strip("\n"), content, flags=re.DOTALL)

with open("convert_echoscene_to_procthor.py", "w") as f:
    f.write(content)

