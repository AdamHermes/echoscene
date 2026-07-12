import prior
import json

try:
    dataset = prior.load_dataset("procthor-10k")
    house = dataset['train'][0]
    with open("sample_house.json", "w") as f:
        json.dump(house, f, indent=2)
    print("Dumped sample_house.json")
except Exception as e:
    print(e)
