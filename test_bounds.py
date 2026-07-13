import json
from ai2thor.controller import Controller

with open("test_procthor_scenes/Bedroom-68718.json", 'r') as f:
    house = json.load(f)

c = Controller(scene="Procedural", agentMode="default", gridSize=0.25, width=300, height=300)
c.reset(scene=house)

print("Scene Bounds:", c.last_event.metadata['sceneBounds'])

fp = house['rooms'][0]['floorPolygon']
print("Floor Polygon:", fp)

c.stop()
