import json

datasets = {
    "all_trainval": "FRONT/relationships_all_trainval.json",
    "all_test": "FRONT/relationships_all_test.json"
}

for name, path in datasets.items():
    with open(path, 'r') as f:
        data = json.load(f)
    scans = [scan['scan'] for scan in data['scans']]
    if "SecondBedroom-6482" in scans:
        print(f"Found in {name} at index {scans.index('SecondBedroom-6482')}")

