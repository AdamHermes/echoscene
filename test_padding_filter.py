import json
from ai2thor.controller import Controller

l, w = 5.0, 5.0

house = {
    "metadata": {"schema": "1.0.0"},
    "proceduralParameters": {},
    "rooms": [{
        "id": "room|0",
        "roomType": "Bedroom",
        "floorMaterial": {"name": "WoodFloorsCross"},
        "floorPolygon": [
            {"x": -2.0, "y": 0, "z": -2.0},
            {"x": -2.0, "y": 0, "z": 7.0},
            {"x": 7.0, "y": 0, "z": 7.0},
            {"x": 7.0, "y": 0, "z": -2.0}
        ],
        "ceilings": []
    }],
    "walls": [
        {"id": "w1", "roomId": "room|0", "polygon": [{"x":0,"y":0,"z":0}, {"x":0,"y":0,"z":5}, {"x":0,"y":3,"z":5}, {"x":0,"y":3,"z":0}]},
        {"id": "w2", "roomId": "room|0", "polygon": [{"x":0,"y":0,"z":5}, {"x":5,"y":0,"z":5}, {"x":5,"y":3,"z":5}, {"x":0,"y":3,"z":5}]},
        {"id": "w3", "roomId": "room|0", "polygon": [{"x":5,"y":0,"z":5}, {"x":5,"y":0,"z":0}, {"x":5,"y":3,"z":0}, {"x":5,"y":3,"z":5}]},
        {"id": "w4", "roomId": "room|0", "polygon": [{"x":5,"y":0,"z":0}, {"x":0,"y":0,"z":0}, {"x":0,"y":3,"z":0}, {"x":5,"y":3,"z":0}]},
        # FAKE INVISIBLE BOUNDARY WALLS TO INFLATE SCENE BOUNDS
        {"id": "bound1", "roomId": "room|0", "polygon": [{"x":-2,"y":0,"z":-2}, {"x":-2,"y":0,"z":-1}, {"x":-2,"y":1,"z":-1}, {"x":-2,"y":1,"z":-2}]},
        {"id": "bound2", "roomId": "room|0", "polygon": [{"x":7,"y":0,"z":7}, {"x":7,"y":0,"z":6}, {"x":7,"y":1,"z":6}, {"x":7,"y":1,"z":7}]}
    ],
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
        all_pts = event.metadata["actionReturn"]
        print(f"Total reachable spots (including outside): {len(all_pts)}")
        
        valid_pts = [p for p in all_pts if 0.0 <= p['x'] <= l and 0.0 <= p['z'] <= w]
        print(f"Valid reachable spots (inside true room): {len(valid_pts)}")
        
        true_walkable = len(valid_pts) * (0.25 ** 2)
        true_total = l * w
        print(f"True Walkability: {true_walkable / true_total:.2%}")
    else:
        print("Failed:", event.metadata["errorMessage"])
except Exception as e:
    print("Error:", e)
finally:
    c.stop()
