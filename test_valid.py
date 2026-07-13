import json
from ai2thor.controller import Controller
data = json.load(open('valid_procthor.json'))

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="Procedural", 
    gridSize=0.25
)
print("Testing valid_procthor.json...")
try:
    event = controller.reset(scene=data)
    event = controller.step(action="GetReachablePositions")
    if event.metadata["lastActionSuccess"]:
        print(f"SUCCESS! Points: {len(event.metadata['actionReturn'])}")
    else:
        print(f"FAILED: {event.metadata['errorMessage']}")
except Exception as e:
    print(f"CRASH: {e}")

