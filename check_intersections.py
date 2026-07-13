import json
import math
from ai2thor.controller import Controller
from shapely.geometry import Polygon, LineString

scene_name = "SecondBedroom-6482"
folder = "baseline"
house_json = f"../output/{folder}/vis/2050/procthor_scenes/{scene_name}.json"

with open(house_json, 'r') as f:
    house_data = json.load(f)

# Offset
rd = house_data.get('room_dims', {})
l = rd.get('l', 0)
w = rd.get('w', 0)

objects_raw = house_data.get('objects_raw', [])
min_x = min(o['position']['x'] - o['size']['x']/2 for o in objects_raw)
max_x = max(o['position']['x'] + o['size']['x']/2 for o in objects_raw)
min_z = min(o['position']['z'] - o['size']['z']/2 for o in objects_raw)
max_z = max(o['position']['z'] + o['size']['z']/2 for o in objects_raw)
fx = (max_x + min_x) / 2.0
fz = (max_z + min_z) / 2.0

offset_x = -(fx - l/2.0)
offset_z = -(fz - w/2.0)

polygons = []
for obj in objects_raw:
    min_y = max(0.0, obj['position']['y'] - obj['size']['y'] / 2.0)
    if min_y >= 1.0: continue
    
    cx = obj['position']['x'] + offset_x
    cz = obj['position']['z'] + offset_z
    obj_w, obj_d = obj['size']['x'], obj['size']['z']
    theta = obj['rotation']['y']
    
    hw, hd = obj_w / 2.0, obj_d / 2.0
    rad = theta
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    
    def rot(lx, lz):
        return cx + lx*cos_t - lz*sin_t, cz + lx*sin_t + lz*cos_t
        
    corners = [rot(-hw, -hd), rot(hw, -hd), rot(hw, hd), rot(-hw, hd)]
    polygons.append(Polygon(corners))

c = Controller(agentMode="default", scene="Procedural", gridSize=0.25)
c.step(action="CreateHouse", house=house_data)
c.step(action="Teleport", position={"x": 2.0, "y": 0.9, "z": 2.0}, forceAction=True)
rp = c.step(action="GetReachablePositions").metadata["actionReturn"]

start_pos = rp[0]
goal_pos = rp[-1]
max_d = 0
for p in rp:
    d = math.dist([start_pos['x'], start_pos['z']], [p['x'], p['z']])
    if d > max_d: max_d = d; goal_pos = p

c.step(action="Teleport", position={"x": start_pos['x'], "y": start_pos['y'], "z": start_pos['z']}, forceAction=True)
path_event = c.step(action="GetShortestPathToPoint", target=goal_pos)
path_points = path_event.metadata["actionReturn"]["corners"]
c.stop()

path_line = LineString([(p['x'], p['z']) for p in path_points])

intersect_count = 0
for i, poly in enumerate(polygons):
    if path_line.intersects(poly):
        print(f"Intersection with Polygon {i}!")
        intersect_count += 1
        
if intersect_count == 0:
    print("NO INTERSECTIONS FOUND! The math perfectly dodges.")
else:
    print(f"FAILED! {intersect_count} intersections.")
