import os

with open('eval_walkability.py', 'r') as f:
    content = f.read()

old_code = """
    # Estimate total floor area based on scene bounds
    # (Assuming a simple rectangular room for now)
    scene_bounds = event.metadata["sceneBounds"]["size"]
    total_area = scene_bounds["x"] * scene_bounds["z"]
"""

new_code = """
    # Calculate total floor area exactly from the room's floorPolygon in the JSON
    # Assuming the room is a rectangle aligned with axes
    floor_poly = scene_data["rooms"][0]["floorPolygon"]
    xs = [p["x"] for p in floor_poly]
    zs = [p["z"] for p in floor_poly]
    l = max(xs) - min(xs)
    w = max(zs) - min(zs)
    total_area = l * w
"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('eval_walkability.py', 'w') as f:
        f.write(content)
    print("Successfully patched eval_walkability.py")
else:
    print("Could not find old code in eval_walkability.py")
