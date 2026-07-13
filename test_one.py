import json
from eval_walkability import calculate_walkability
from ai2thor.controller import Controller

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="Procedural", 
    gridSize=0.25
)

score, walkable, total, points = calculate_walkability("../output/physcene_guidance/vis/2050/procthor_scenes/MasterBedroom-18415.json", controller)
print(f"Score: {score:.2%} ({walkable:.2f}m² / {total:.2f}m², Points: {points})")
