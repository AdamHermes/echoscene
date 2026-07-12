import json
from deepdiff import DeepDiff

with open("sample_house.json", "r") as f:
    sample = json.load(f)

with open("../output/released_full_model/vis/2050/procthor_scenes/SecondBedroom-30034.json", "r") as f:
    mine = json.load(f)

# Ignore objects list, we just want to compare the house top-level and first room/wall
sample['objects'] = []
mine['objects'] = []
sample['doors'] = []
mine['doors'] = []
sample['windows'] = []
mine['windows'] = []

# Keep only 1 room and 1 wall for comparison
if len(sample['rooms']) > 1: sample['rooms'] = [sample['rooms'][0]]
if len(sample['walls']) > 1: sample['walls'] = [sample['walls'][0]]

if len(mine['rooms']) > 1: mine['rooms'] = [mine['rooms'][0]]
if len(mine['walls']) > 1: mine['walls'] = [mine['walls'][0]]

diff = DeepDiff(sample, mine, ignore_order=True)
print(diff.pretty())
