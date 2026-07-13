import json
from ai2thor.controller import Controller

controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)
with open("../output/baseline/vis/2050/procthor_scenes/SecondBedroom-30034.json", "r") as f:
    house_data = json.load(f)

event = controller.reset(scene=house_data)
if 'sceneBounds' in event.metadata:
    print("Scene Bounds:", event.metadata['sceneBounds'])
else:
    print("No sceneBounds in metadata")

tp_event = controller.step(action="Teleport", position={"x": 0.25, "y": 0.9, "z": 0.25}, forceAction=True)
print("Teleport:", tp_event.metadata['lastActionSuccess'])
print("Teleport pos:", tp_event.metadata['agent']['position'])

rp_event = controller.step(action="GetReachablePositions")
print("RP Success:", rp_event.metadata['lastActionSuccess'])
if not rp_event.metadata['lastActionSuccess']:
    print("Error:", rp_event.metadata['errorMessage'])

controller.stop()
