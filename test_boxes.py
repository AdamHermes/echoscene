import json

with open('./output/baseline/vis/2050/procthor_scenes/SecondBedroom-6482.json') as f:
    data = json.load(f)

print("FURNITURE BOXES:")
for b in data.get('furniture_boxes', []):
    print(b)

print("\nOBJECTS RAW:")
for o in data.get('objects_raw', []):
    print(o['position'], o['size'], o['rotation'])

