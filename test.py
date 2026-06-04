import json

with open('./FRONT/relationships_all_test.json') as f:
    data = json.load(f)

room_dict = {
    'bedroom': ['Bedroom', 'MasterBedroom', 'SecondBedroom'],
    'livingroom': ['LivingDiningRoom', 'LivingRoom'],
    'diningroom': ['LivingDiningRoom', 'DiningRoom']
}

for i, scan in enumerate(data['scans']):
    room_name = scan['scan'].split('-')[0]
    for room_type, prefixes in room_dict.items():
        if room_name in prefixes:
            print(f'idx={i:3d} | {room_type:12s} | {scan["scan"]}')
            break
            
