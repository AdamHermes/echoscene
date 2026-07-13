import sys

with open("convert_echoscene_to_procthor.py", "r") as f:
    content = f.read()

old_code = """            # clamp to room bounds to avoid walls leaking outside
            wx = max(0.01, min(cx + rx, l - 0.01))
            wz = max(0.01, min(cz + rz, w - 0.01))
            world_corners.append((wx, wz))
            
        shifted_obj_polys.append(shapely.geometry.Polygon(world_corners))"""

new_code = """            world_corners.append((cx + rx, cz + rz))
            
        poly = shapely.geometry.Polygon(world_corners)
        # Room bounds with a tiny 0.01m inset to ensure nothing leaks into walls
        room_poly = shapely.geometry.Polygon([(0.01, 0.01), (0.01, w-0.01), (l-0.01, w-0.01), (l-0.01, 0.01)])
        clipped_poly = poly.intersection(room_poly)
        if not clipped_poly.is_empty:
            shifted_obj_polys.append(clipped_poly)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("convert_echoscene_to_procthor.py", "w") as f:
        f.write(content)
    print("Successfully patched.")
else:
    print("Could not find old code.")
