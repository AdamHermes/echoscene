import re

with open("convert_echoscene_to_procthor.py", "r") as f:
    content = f.read()

# Replace y: 0.9 with y: 1.5
content = content.replace('"y": 0.9', '"y": 1.5')

with open("convert_echoscene_to_procthor.py", "w") as f:
    f.write(content)

