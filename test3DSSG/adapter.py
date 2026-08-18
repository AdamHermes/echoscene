import json
import os

# 1. TỪ ĐIỂN MAPPING CLASS (3DSSG Label -> EchoScene Label)
# Map nhãn thực tế của 3DSSG sang nhãn của EchoScene (SG-FRONT)
CLASS_MAPPING = {
    "bed": "bed",
    "chair": "chair",
    "armchair": "chair",
    "sofa": "sofa",
    "couch": "sofa",
    "table": "table",
    "dining table": "table",
    "coffee table": "table",
    "desk": "desk",
    "nightstand": "nightstand",
    "cabinet": "cabinet",
    "wardrobe": "cabinet",
    "bookshelf": "shelf",
    "shelf": "shelf",
    "lamp": "lamp",
    "tv stand": "cabinet",
    "tv": "tv",
    "stool": "chair",
    "bench": "chair"
}

# 2. TỪ ĐIỂN MAPPING RELATIONSHIPS
REL_MAPPING = {
    "left": "left",
    "right": "right",
    "front": "front",
    "behind": "behind",
    "standing on": "standing on",
    "close by": "close by",
    "supported by": "standing on",
    "attached to": "close by",
    "hanging on": "close by",
    "lying on": "standing on"
}

def create_echoscene_adapter(objects_file, rels_file, scan_id, output_file):
    print(f"Reading data from scan: {scan_id}...")
    # Load 3DSSG Data
    with open(objects_file, 'r') as f:
        objects_data = json.load(f)
    with open(rels_file, 'r') as f:
        rels_data = json.load(f)
        
    # Tìm scan cụ thể
    try:
        scan_objects = next(s for s in objects_data['scans'] if s['scan'] == scan_id)['objects']
        scan_rels = next(s for s in rels_data['scans'] if s['scan'] == scan_id)['relationships']
    except StopIteration:
        print(f"Error: Could not find scan_id '{scan_id}' in 3DSSG data.")
        return

    # LỌC VÀ CHUYỂN ĐỔI NODES
    valid_nodes = {} # id -> mapped_class
    echoscene_nodes = []
    
    for obj in scan_objects:
        raw_label = obj.get('label', '').lower()
        if raw_label in CLASS_MAPPING:
            mapped_class = CLASS_MAPPING[raw_label]
            valid_nodes[str(obj['id'])] = mapped_class
            echoscene_nodes.append({
                "id": str(obj['id']),
                "class_label": mapped_class,
                "original_label": raw_label
            })
            
    # LỌC VÀ CHUYỂN ĐỔI EDGES
    echoscene_edges = []
    for rel in scan_rels:
        obj1_id, obj2_id, rel_id, rel_name = rel
        obj1_id = str(obj1_id)
        obj2_id = str(obj2_id)
        
        # Chỉ giữ lại edge nếu CẢ 2 object đều nằm trong danh sách valid_nodes
        if obj1_id in valid_nodes and obj2_id in valid_nodes:
            mapped_rel = REL_MAPPING.get(rel_name.lower())
            if mapped_rel:
                echoscene_edges.append([obj1_id, mapped_rel, obj2_id])

    # ĐÓNG GÓI OUTPUT THEO FORMAT CỦA ECHOSCENE
    adapted_scene = {
        "scene_id": f"adapted_3dssg_{scan_id}",
        "room_type": "livingroom", # Giả định room type cho EchoScene
        "room_size": [6.0, 3.0, 5.0], # Dummy room size: X, Y, Z (mét) - EchoScene cần
        "nodes": echoscene_nodes,
        "edges": echoscene_edges
    }

    # Đảm bảo thư mục output tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Lưu ra file mới
    with open(output_file, 'w') as f:
        json.dump(adapted_scene, f, indent=4)
        
    print(f"Conversion complete!")
    print(f"Objects kept: {len(echoscene_nodes)}")
    print(f"Relationships kept: {len(echoscene_edges)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Đường dẫn tới thư mục gốc của project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    OBJECTS_FILE = os.path.join(BASE_DIR, "3DSSG", "objects.json")
    RELS_FILE = os.path.join(BASE_DIR, "3DSSG", "relationships.json")
    
    # Một scan_id bất kỳ trong 3DSSG (scan đầu tiên trong objects.json)
    TARGET_SCAN_ID = "0988ea72-eb32-2e61-8344-99e2283c2728" 
    
    OUTPUT_FILE = os.path.join(BASE_DIR, "test3DSSG", f"adapted_{TARGET_SCAN_ID}.json")
    
    create_echoscene_adapter(OBJECTS_FILE, RELS_FILE, TARGET_SCAN_ID, OUTPUT_FILE)
