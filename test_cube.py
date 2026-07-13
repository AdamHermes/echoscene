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
            {"x": 0.0, "y": 0, "z": 0.0},
            {"x": 0.0, "y": 0, "z": 5.0},
            {"x": 5.0, "y": 0, "z": 5.0},
            {"x": 5.0, "y": 0, "z": 0.0}
        ],
        "ceilings": []
    }],
    "walls": [],
    "objects": [
        {
            "id": "test_cube",
            "assetId": "Chair_1",
            "position": {"x": 2.5, "y": 0.5, "z": 2.5},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "scale": {"x": 2.0, "y": 1.0, "z": 1.5},
            "kinematic": True
        }
    ],
    "doors": [],
    "windows": []
}

c = Controller(scene="Procedural", agentMode="default", gridSize=0.25, width=300, height=300)
try:
    c.reset(scene=house)
    print("Successfully spawned house with Cube asset!")
    print("Objects in scene:", len(c.last_event.metadata['objects']))
    if len(c.last_event.metadata['objects']) > 0:
        print("First object name:", c.last_event.metadata['objects'][0]['name'])
    
    event = c.step(action="GetReachablePositions")
    if event.metadata["lastActionSuccess"]:
        print("Walkable spots:", len(event.metadata["actionReturn"]))
    else:
        print("Failed to get reachable:", event.metadata["errorMessage"])
except Exception as e:
    print("Error:", e)
finally:
    c.stop()
