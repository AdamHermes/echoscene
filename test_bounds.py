import ai2thor.controller
import json

import os

scenes_dir = "../output/released_full_model/vis/2050/procthor_scenes"
scene_file = next(f for f in os.listdir(scenes_dir) if f.endswith(".json") and f != "walkability_results.json")
with open(os.path.join(scenes_dir, scene_file), "r") as f:
    house = json.load(f)
    print(f"Loaded {scene_file}")

controller = ai2thor.controller.Controller(
    scene=house,
    agentMode="default",
    visibilityDistance=1.5,
    fieldOfView=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
    gridSize=0.25
)

evt = controller.step(action="GetSceneBounds")
if evt.metadata["lastActionSuccess"]:
    print("Scene Bounds:", evt.metadata["actionReturn"])
else:
    print("Failed to get scene bounds:", evt.metadata["errorMessage"])

evt = controller.step(action="GetReachablePositions")
if not evt.metadata["lastActionSuccess"]:
    print("GetReachablePositions error:", evt.metadata["errorMessage"])
