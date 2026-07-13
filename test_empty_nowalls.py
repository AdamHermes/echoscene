import json
from ai2thor.controller import Controller

house = {
    "metadata": {"schema": "1.0.0"},
    "proceduralParameters": {},
    "rooms": [{
        "id": "room|0",
        "roomType": "Bedroom",
        "floorMaterial": {"name": "WoodFloorsCross"},
        "floorPolygon": [
            {"x": -0.1, "y": 0, "z": -0.1},
            {"x": -0.1, "y": 0, "z": 5.1},
            {"x": 5.1, "y": 0, "z": 5.1},
            {"x": 5.1, "y": 0, "z": -0.1}
        ],
        "ceilings": []
    }],
    "walls": [],
    "objects": [],
    "doors": [],
    "windows": []
}

c = Controller(scene="Procedural", agentMode="default", gridSize=0.25, width=300, height=300)
try:
    c.reset(scene=house)
    c.step(action="Teleport", position={"x": 2.5, "y": 0.9, "z": 2.5})
    event = c.step(action="GetReachablePositions")
    if event.metadata["lastActionSuccess"]:
        print("Walkable spots:", len(event.metadata["actionReturn"]))
    else:
        print("Failed:", event.metadata["errorMessage"])
except Exception as e:
    print("Error:", e)
finally:
    c.stop()
