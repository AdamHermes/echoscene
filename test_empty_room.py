import json
import traceback
from ai2thor.controller import Controller

empty_house = {
    "metadata": {"schema": "1.0.0"},
    "proceduralParameters": {
        "ceilingColor": {"b": 1.0, "g": 1.0, "r": 1.0},
        "ceilingMaterial": {"name": "PureWhite", "color": {"b": 1.0, "g": 1.0, "r": 1.0}},
        "floorColliderThickness": 1.0,
        "lights": [],
        "receptacleHeight": 0.7,
        "reflections": [],
        "skyboxId": "SkyAlbany"
    },
    "rooms": [{
        "id": "room|0",
        "roomType": "Bedroom",
        "children": [],
        "floorMaterial": {"name": "WoodFloorsCross", "color": {"r": 1.0, "g": 1.0, "b": 1.0}},
        "floorPolygon": [
            {"x": 0.0, "y": 0, "z": 0.0},
            {"x": 0.0, "y": 0, "z": 4.0},
            {"x": 4.0, "y": 0, "z": 4.0},
            {"x": 4.0, "y": 0, "z": 0.0}
        ],
        "ceilings": []
    }],
    "walls": [
        {
            "id": "wall|0|0.0|0.0|0.0|4.0",
            "roomId": "room|0",
            "color": {"r": 1.0, "g": 1.0, "b": 1.0},
            "material": {"name": "PureWhite", "color": {"r": 1.0, "g": 1.0, "b": 1.0}},
            "polygon": [
                {"x": 0.0, "y": 0, "z": 0.0},
                {"x": 0.0, "y": 0, "z": 4.0},
                {"x": 0.0, "y": 3.0, "z": 0.0},
                {"x": 0.0, "y": 3.0, "z": 4.0}
            ]
        },
        {
            "id": "wall|0|0.0|4.0|4.0|4.0",
            "roomId": "room|0",
            "color": {"r": 1.0, "g": 1.0, "b": 1.0},
            "material": {"name": "PureWhite", "color": {"r": 1.0, "g": 1.0, "b": 1.0}},
            "polygon": [
                {"x": 0.0, "y": 0, "z": 4.0},
                {"x": 4.0, "y": 0, "z": 4.0},
                {"x": 0.0, "y": 3.0, "z": 4.0},
                {"x": 4.0, "y": 3.0, "z": 4.0}
            ]
        },
        {
            "id": "wall|0|4.0|4.0|4.0|0.0",
            "roomId": "room|0",
            "color": {"r": 1.0, "g": 1.0, "b": 1.0},
            "material": {"name": "PureWhite", "color": {"r": 1.0, "g": 1.0, "b": 1.0}},
            "polygon": [
                {"x": 4.0, "y": 0, "z": 4.0},
                {"x": 4.0, "y": 0, "z": 0.0},
                {"x": 4.0, "y": 3.0, "z": 4.0},
                {"x": 4.0, "y": 3.0, "z": 0.0}
            ]
        },
        {
            "id": "wall|0|4.0|0.0|0.0|0.0",
            "roomId": "room|0",
            "color": {"r": 1.0, "g": 1.0, "b": 1.0},
            "material": {"name": "PureWhite", "color": {"r": 1.0, "g": 1.0, "b": 1.0}},
            "polygon": [
                {"x": 4.0, "y": 0, "z": 0.0},
                {"x": 0.0, "y": 0, "z": 0.0},
                {"x": 4.0, "y": 3.0, "z": 0.0},
                {"x": 0.0, "y": 3.0, "z": 0.0}
            ]
        }
    ],
    "objects": [],
    "windows": [],
    "doors": []
}

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="Procedural", 
    gridSize=0.25
)

def test_spawn(x, y, z):
    agent_info = {
        "spawn_x": x,
        "spawn_z": z,
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "standing": True,
        "horizon": 30
    }
    empty_house["metadata"]["agent"] = agent_info
    empty_house["metadata"]["agentPoses"] = {
        "default": agent_info,
        "locobot": agent_info,
        "stretch": agent_info
    }
    
    try:
        event = controller.reset(scene=empty_house)
        event = controller.step(action="GetReachablePositions")
        if event.metadata["lastActionSuccess"]:
            print(f"SUCCESS at {x}, {y}, {z}! Points: {len(event.metadata['actionReturn'])}")
            print(f"Scene Bounds: {event.metadata['sceneBounds']['size']}")
        else:
            print(f"FAILED at {x}, {y}, {z}: {event.metadata['errorMessage']}")
    except Exception as e:
        print(f"CRASH at {x}, {y}, {z}: {e}")

test_spawn(2.0, 0.9, 2.0)

