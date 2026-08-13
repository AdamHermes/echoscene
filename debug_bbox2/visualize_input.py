import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

def get_obb_corners(x, z, l, w, angle_rad):
    """Calculates the 4 corners of the Oriented Bounding Box (OBB)."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append([x + rx, z + rz])
    return np.array(rotated)

def main():
    target_scene = "DiningRoom-31158"
    
    # Path setup
    base_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    rel_files = [
        os.path.join(base_dir, "FRONT/relationships_diningroom_test.json"),
        os.path.join(base_dir, "FRONT/relationships_all_test.json"),
        os.path.join(base_dir, "FRONT/relationships_bedroom_test.json")
    ]
    box_files = [
        os.path.join(base_dir, "FRONT/obj_boxes_diningroom_test.json"),
        os.path.join(base_dir, "FRONT/obj_boxes_all_test.json"),
        os.path.join(base_dir, "FRONT/obj_boxes_bedroom_test.json")
    ]
    
    # Artifact output directory to share with user
    artifact_dir = "/Users/lehoangan/.gemini/antigravity-cli/brain/c05e6e51-8949-4274-95b9-1f2ea3fe5868"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Load relationships and objects
    scene_rel = None
    for rf in rel_files:
        if os.path.exists(rf):
            with open(rf, 'r') as f:
                rel_data = json.load(f)
                for scan in rel_data.get('scans', []):
                    if scan['scan'] == target_scene:
                        scene_rel = scan
                        break
        if scene_rel:
            break
            
    if not scene_rel:
        print(f"Scene {target_scene} not found in relationship files!")
        return
        
    # Simplify class names
    objects_map = {}
    raw_class_map = {}
    for k, v in scene_rel['objects'].items():
        raw_class_map[k] = v
        name = v.lower()
        if 'bed' in name:
            name = 'bed'
        elif 'chair' in name or 'armchair' in name:
            name = 'chair'
        elif 'table' in name or 'desk' in name:
            name = 'table'
        elif 'lamp' in name:
            name = 'lamp'
        elif 'floor' in name:
            name = 'floor'
        objects_map[k] = name

    # 2. Load bounding boxes
    box_data = None
    for bf in box_files:
        if os.path.exists(bf):
            with open(bf, 'r') as f:
                data = json.load(f)
                if target_scene in data:
                    box_data = data
                    break
                    
    if not box_data or target_scene not in box_data:
        print(f"Scene {target_scene} not found in bounding box data!")
        return
        
    scene_center = np.array(box_data[target_scene]["scene_center"])
    
    # Color map for 2D layout and graph nodes
    color_map = {
        "bed": "#3498db", "chair": "#e74c3c", "nightstand": "#9b59b6", 
        "table": "#e67e22", "wardrobe": "#2ecc71", "lamp": "#f1c40f", 
        "floor": "#bdc3c7"
    }
    
    # ==========================================
    # VISUALIZATION 1: 2D Bounding Box Layout
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    all_xs = []
    all_zs = []
    
    for k, name in objects_map.items():
        if k not in box_data[target_scene]:
            continue
        param7 = box_data[target_scene][k]["param7"]
        l, h, w = param7[0:3]
        x, y, z = param7[3:6]
        angle_rad = param7[6]
        
        # Center subtraction
        x -= scene_center[0]
        z -= scene_center[2]
        
        obb_corners = get_obb_corners(x, z, l, w, angle_rad)
        all_xs.extend(obb_corners[:, 0])
        all_zs.extend(obb_corners[:, 1])
        
        color = color_map.get(name, "#1abc9c")
        alpha = 0.25 if name == "floor" else 0.7
        zorder = 1 if name == "floor" else 10
        
        # Draw Oriented Bounding Box
        polygon = patches.Polygon(
            obb_corners, closed=True, facecolor=color,
            edgecolor='black', alpha=alpha, linewidth=1.5, zorder=zorder
        )
        ax.add_patch(polygon)
        
        # Label (except floor)
        if name != "floor":
            ax.text(x, z, name, ha='center', va='center',
                    fontsize=10, weight='bold', zorder=zorder+1,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                    
    # Adjust plot limits
    ax.set_aspect('equal')
    if all_xs:
        margin = 0.5
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_zs) - margin, max(all_zs) + margin)
    else:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title(f"Original Input Layout: {target_scene}\n(From 3D-FRONT Dataset)", fontsize=12, weight='bold', pad=10)
    
    # Save Layout Plots
    layout_local_path = os.path.join(script_dir, f"{target_scene}_input_layout.png")
    layout_artifact_path = os.path.join(artifact_dir, f"{target_scene}_input_layout.png")
    plt.savefig(layout_local_path, bbox_inches='tight', dpi=150)
    plt.savefig(layout_artifact_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved layout visualization to {layout_local_path}")
    
    # ==========================================
    # VISUALIZATION 2: Scene Graph (Relationships)
    # ==========================================
    G = nx.DiGraph()
    
    # Define node label maps
    node_labels = {}
    node_colors_list = []
    
    # Add nodes
    for k, name in objects_map.items():
        node_id = f"{name}_{k}"
        node_labels[node_id] = f"{name} ({k})"
        G.add_node(node_id)
        
    # Spatial relations to focus on
    spatial_relations = ['left', 'right', 'front', 'behind', 'above', 'close by', 'standing on']
    
    for rel in scene_rel['relationships']:
        src, dest, rel_idx, rel_type = rel
        src_str = str(src)
        dest_str = str(dest)
        
        if src_str in objects_map and dest_str in objects_map:
            # Skip "standing on floor" since everything stands on it
            if rel_type == 'standing on' and objects_map[dest_str] == 'floor':
                continue
                
            src_node = f"{objects_map[src_str]}_{src_str}"
            dest_node = f"{objects_map[dest_str]}_{dest_str}"
            
            if rel_type in spatial_relations:
                G.add_edge(src_node, dest_node, label=rel_type)
                
    # Prepare node colors
    for node in G.nodes():
        name = node.split('_')[0]
        node_colors_list.append(color_map.get(name, "#1abc9c"))
        
    # Plot graph
    plt.figure(figsize=(10, 8))
    
    # Choose a layout
    pos = nx.shell_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors_list, edgecolors='black', linewidths=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family="sans-serif", font_weight="bold")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.5, arrowstyle='-|>', arrowsize=20, edge_color='#7f8c8d')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#c0392b', font_weight='bold')
    
    plt.title(f"Scene Graph Spatial Relationships: {target_scene}", fontsize=12, weight='bold', pad=10)
    plt.axis('off')
    
    # Save Graph Plots
    graph_local_path = os.path.join(script_dir, f"{target_scene}_input_graph.png")
    graph_artifact_path = os.path.join(artifact_dir, f"{target_scene}_input_graph.png")
    plt.savefig(graph_local_path, bbox_inches='tight', dpi=150)
    plt.savefig(graph_artifact_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved scene graph visualization to {graph_local_path}")

if __name__ == "__main__":
    main()
