import json

datasets = {
    "bedroom_trainval": "FRONT/relationships_bedroom_trainval.json",
    "bedroom_test": "FRONT/relationships_bedroom_test.json"
}

for name, path in datasets.items():
    with open(path, 'r') as f:
        data = json.load(f)
    scans = [scan['scan'] for scan in data['scans']]
    if "SecondBedroom-6482" in scans:
        print(f"Found in {name} at index {scans.index('SecondBedroom-6482')}")

