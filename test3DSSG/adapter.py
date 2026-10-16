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

def convert_all_scenes():
    # Đường dẫn tới thư mục gốc của project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    OBJECTS_FILE = os.path.join(BASE_DIR, "3DSSG", "objects.json")
    RELS_FILE = os.path.join(BASE_DIR, "3DSSG", "relationships.json")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "test3DSSG", "adapted_scenes")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Reading 3DSSG Data...")
    with open(OBJECTS_FILE, 'r') as f:
        objects_data = json.load(f)
    with open(RELS_FILE, 'r') as f:
        rels_data = json.load(f)
        
    rels_dict = {s['scan']: s['relationships'] for s in rels_data['scans']}
    
    total_scans = len(objects_data['scans'])
    success_count = 0
    
    for scan in objects_data['scans']:
        scan_id = scan['scan']
        # Dữ liệu 3DSSG không cung cấp room_type ở cấp độ scene, ta mặc định là "all" để tương thích EchoScene
        room_type = "all"
        
        scan_objects = scan.get('objects', [])
        scan_rels = rels_dict.get(scan_id, [])
        
        # LỌC NODES
        valid_nodes = {}
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
                
        # LỌC EDGES
        echoscene_edges = []
        for rel in scan_rels:
            obj1_id, obj2_id, rel_id, rel_name = rel
            obj1_id = str(obj1_id)
            obj2_id = str(obj2_id)
            if obj1_id in valid_nodes and obj2_id in valid_nodes:
                mapped_rel = REL_MAPPING.get(rel_name.lower())
                if mapped_rel:
                    echoscene_edges.append([obj1_id, mapped_rel, obj2_id])
                    
        # Nếu scene sau khi lọc không còn đồ vật nào, bỏ qua
        if len(echoscene_nodes) == 0:
            continue
            
        adapted_scene = {
            "scene_id": f"adapted_3dssg_{scan_id}",
            "room_type": room_type,
            "room_size": [6.0, 3.0, 5.0],
            "nodes": echoscene_nodes,
            "edges": echoscene_edges
        }
        
        output_file = os.path.join(OUTPUT_DIR, f"adapted_{scan_id}.json")
        with open(output_file, 'w') as f:
            json.dump(adapted_scene, f, indent=4)
        success_count += 1
        
    print(f"Processed {total_scans} scans. Successfully created {success_count} adapted scenes (others were empty after filtering).")
    print(f"All files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_all_scenes()
