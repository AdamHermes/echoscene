import json
import math
import numpy as np

def is_clear(x, z, furniture_boxes, l, w):
    clearance = 0.3
    for box in furniture_boxes:
        min_x, max_x, min_z, max_z = box
        if (min_x - clearance <= x <= max_x + clearance) and (min_z - clearance <= z <= max_z + clearance):
            return False
    if x < clearance or x > l - clearance or z < clearance or z > w - clearance:
        return False
    return True

scene_name = "SecondBedroom-6482"
folder = "baseline"
house_json = f"./output/{folder}/vis/2050/procthor_scenes/{scene_name}.json"

with open(house_json, 'r') as f:
    house_data = json.load(f)

rd = house_data.get('room_dims', {})
l = rd.get('l', 0)
w = rd.get('w', 0)
furniture_boxes = house_data.get('furniture_boxes', [])

spawn_x, spawn_z = l / 2.0, w / 2.0
found = False
for x in np.arange(0.25, l, 0.25):
    for z in np.arange(0.25, w, 0.25):
        if is_clear(x, z, furniture_boxes, l, w):
            spawn_x, spawn_z = x, z
            found = True
            break
    if found: break

print(f"plot_path.py spawn point is: {spawn_x}, {spawn_z}")
