import re

with open("convert_echoscene_to_procthor.py", "r") as f:
    content = f.read()

new_logic = """
    if "metadata" not in house:
        house["metadata"] = {}
        
    agent_info = {
        "spawn_x": best_spawn[0],
        "spawn_z": best_spawn[1],
        "position": {"x": best_spawn[0], "y": 0.9, "z": best_spawn[1]},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "standing": True,
        "horizon": 30
    }
    house["metadata"]["agent"] = agent_info
    house["metadata"]["agentPoses"] = {
        "default": agent_info,
        "locobot": agent_info,
        "stretch": agent_info
    }
    
    return house
"""

pattern = r'    if "metadata" not in house:.*?return house'
content = re.sub(pattern, new_logic.strip("\n"), content, flags=re.DOTALL)

with open("convert_echoscene_to_procthor.py", "w") as f:
    f.write(content)

