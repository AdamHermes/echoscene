import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import plotly.graph_objects as go

def main():
    target_scene = "SecondBedroom-6482"
    base_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene"
    rel_file = os.path.join(base_dir, "FRONT/relationships_bedroom_test.json")
    box_file = os.path.join(base_dir, "FRONT/obj_boxes_bedroom_test.json")
    
    # Destination directories
    debug_bbox_dir = os.path.join(base_dir, "debug_bbox")
    artifact_dir = "/Users/lehoangan/.gemini/antigravity-cli/brain/4e0ee128-7f37-4582-bf1c-d49a9b96abc4"
    
    os.makedirs(debug_bbox_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Load Ground Truth Data
    with open(rel_file, 'r') as f:
        rel_scans = json.load(f)['scans']
        scene_rel = next(s for s in rel_scans if s['scan'] == target_scene)
        
    with open(box_file, 'r') as f:
        box_data = json.load(f)[target_scene]
        
    scene_center = np.array(box_data['scene_center'])
    
    # Object Category Mapping
    obj_id_to_cat = scene_rel['objects'] # e.g. {"1": "double_bed", "2": "armchair", ...}
    
    # Color Map
    color_map = {
        "double_bed": "#3498db",   # Blue
        "bed": "#3498db",
        "armchair": "#e74c3c",     # Red
        "chair": "#e74c3c",
        "nightstand": "#9b59b6",   # Purple
        "table": "#e67e22",        # Orange
        "wardrobe": "#2ecc71",     # Green
        "ceiling_lamp": "#f1c40f", # Yellow
        "lamp": "#f1c40f",
        "floor": "#bdc3c7"         # Gray
    }

    # Extract parsed objects list
    objects = []
    for obj_id, cat_name in obj_id_to_cat.items():
        if obj_id not in box_data:
            continue
        obj_raw = box_data[obj_id]
        p7 = obj_raw['param7']
        l, h, w, gx, gy, gz, angle = p7
        pts_global = np.array(obj_raw['8points'])
        pts_rel = pts_global - scene_center
        
        rx = gx - scene_center[0]
        ry = gy - scene_center[1]
        rz = gz - scene_center[2]
        
        objects.append({
            "id": obj_id,
            "category": cat_name,
            "l": l, "h": h, "w": w,
            "x_rel": rx, "y_rel": ry, "z_rel": rz,
            "x_global": gx, "y_global": gy, "z_global": gz,
            "angle_rad": angle,
            "angle_deg": np.degrees(angle),
            "8points_rel": pts_rel,
            "8points_global": pts_global,
            "model_path": obj_raw.get("model_path"),
            "color": color_map.get(cat_name, "#1abc9c")
        })

    print(f"Loaded {len(objects)} ground truth objects for {target_scene}.")

    # =========================================================================
    # VISUALIZATION 1: 2D Floorplan Top-Down Layout (X-Z plane)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 9))
    
    all_xs, all_zs = [], []
    
    for obj in objects:
        pts = obj["8points_rel"]
        # pts has 8 corners, top-down X-Z projection is bottom 4 corners: 0, 1, 5, 4
        # or min/max corners of the rotated OBB
        # Let's extract X-Z coordinates from all 8 points for polygon hull
        xz_pts = pts[:, [0, 2]]
        all_xs.extend(xz_pts[:, 0])
        all_zs.extend(xz_pts[:, 1])
        
        # Calculate OBB corners for top-down
        # Order of bottom 4 vertices: 0 -> 1 -> 5 -> 4
        poly_corners = pts[[0, 1, 5, 4], :][:, [0, 2]]
        
        alpha = 0.25 if obj["category"] == "floor" else 0.75
        zorder = 1 if obj["category"] == "floor" else 10
        
        polygon = patches.Polygon(
            poly_corners, closed=True, facecolor=obj["color"],
            edgecolor='black', alpha=alpha, linewidth=1.5, zorder=zorder
        )
        ax.add_patch(polygon)
        
        # Label & Orientation Arrow
        if obj["category"] != "floor":
            # Center of top-down bounding box
            cx, cz = obj["x_rel"], obj["z_rel"]
            ax.text(cx, cz, f"{obj['category']}\n(ID {obj['id']})", ha='center', va='center',
                    fontsize=9, weight='bold', zorder=zorder+2,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2, boxstyle='round,pad=0.3'))
            
            # Orientation arrow showing front direction (+X local rotated by angle)
            angle_rad = obj["angle_rad"]
            arrow_len = 0.4
            dx = arrow_len * np.cos(angle_rad)
            dz = arrow_len * np.sin(angle_rad)
            ax.annotate("", xy=(cx + dx, cz + dz), xytext=(cx, cz),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=zorder+3)

    ax.set_aspect('equal')
    margin = 0.6
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_zs) - margin, max(all_zs) + margin)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("X (meters)", fontsize=11, weight='bold')
    ax.set_ylabel("Z (meters, Depth)", fontsize=11, weight='bold')
    ax.set_title(f"3D-FRONT Ground Truth 2D Layout: {target_scene}\n(Top-Down X-Z View)", fontsize=13, weight='bold', pad=12)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor=obj["color"], edgecolor='black', label=f"{obj['category']} (ID {obj['id']})")
        for obj in objects if obj["category"] != "floor"
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    layout_local_path = os.path.join(debug_bbox_dir, f"{target_scene}_GT_2D_Layout.png")
    layout_artifact_path = os.path.join(artifact_dir, f"{target_scene}_GT_2D_Layout.png")
    plt.savefig(layout_local_path, bbox_inches='tight', dpi=200)
    plt.savefig(layout_artifact_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved 2D Layout to {layout_local_path}")

    # =========================================================================
    # VISUALIZATION 2: 3D Perspective & Multi-View Bounding Boxes (Matplotlib 3D)
    # =========================================================================
    fig = plt.figure(figsize=(14, 12))
    
    # Define box 12 edges
    edges = [
        (0, 1), (1, 5), (5, 4), (4, 0),  # Bottom
        (2, 3), (3, 7), (7, 6), (6, 2),  # Top
        (0, 2), (1, 3), (4, 6), (5, 7)   # Vertical pillars
    ]
    
    # Define box 6 faces for translucent shading
    faces_indices = [
        [0, 1, 5, 4], # Bottom
        [2, 3, 7, 6], # Top
        [0, 1, 3, 2], # Front
        [4, 5, 7, 6], # Back
        [0, 4, 6, 2], # Left
        [1, 5, 7, 3]  # Right
    ]
    
    # Subplot 1: Perspective Isometric View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for obj in objects:
        pts = obj["8points_rel"]
        color = obj["color"]
        alpha = 0.12 if obj["category"] == "floor" else 0.35
        
        # Draw 6 faces
        face_verts = [[pts[idx] for idx in face] for face in faces_indices]
        poly = Poly3DCollection(face_verts, facecolors=color, edgecolors='black', linewidths=1.2, alpha=alpha)
        ax1.add_collection3d(poly)
        
        # Draw wireframe edges
        for e1, e2 in edges:
            ax1.plot3D([pts[e1, 0], pts[e2, 0]],
                       [pts[e1, 2], pts[e2, 2]],
                       [pts[e1, 1], pts[e2, 1]], color='black', lw=1.2)
            
        # Label
        if obj["category"] != "floor":
            ax1.text(obj["x_rel"], obj["z_rel"], obj["y_rel"] + obj["h"]/2,
                     f"{obj['category']}", color='black', weight='bold', fontsize=8,
                     ha='center', va='center')
                     
    ax1.set_xlabel("X (m)", weight='bold')
    ax1.set_ylabel("Z (Depth m)", weight='bold')
    ax1.set_zlabel("Y (Height m)", weight='bold')
    ax1.set_title("3D Isometric Perspective View", weight='bold', fontsize=11)
    ax1.view_init(elev=28, azim=-55)

    # Subplot 2: Top View (X-Z)
    ax2 = fig.add_subplot(2, 2, 2)
    for obj in objects:
        pts = obj["8points_rel"]
        poly_corners = pts[[0, 1, 5, 4], :][:, [0, 2]]
        polygon = patches.Polygon(poly_corners, closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.6, lw=1.2)
        ax2.add_patch(polygon)
        if obj["category"] != "floor":
            ax2.text(obj["x_rel"], obj["z_rel"], obj["category"], ha='center', va='center', fontsize=8, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax2.set_aspect('equal')
    ax2.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax2.set_ylim(min(all_zs) - margin, max(all_zs) + margin)
    ax2.set_xlabel("X (m)", weight='bold')
    ax2.set_ylabel("Z (Depth m)", weight='bold')
    ax2.set_title("Top View (X-Z Plane)", weight='bold', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Subplot 3: Front View (X-Y)
    ax3 = fig.add_subplot(2, 2, 3)
    all_ys = [p[1] for obj in objects for p in obj["8points_rel"]]
    for obj in objects:
        pts = obj["8points_rel"]
        poly_corners = pts[[0, 1, 3, 2], :][:, [0, 1]]
        polygon = patches.Polygon(poly_corners, closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.6, lw=1.2)
        ax3.add_patch(polygon)
        if obj["category"] != "floor":
            ax3.text(obj["x_rel"], obj["y_rel"] + obj["h"]/2, obj["category"], ha='center', va='center', fontsize=8, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax3.set_aspect('equal')
    ax3.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax3.set_ylim(min(all_ys) - 0.2, max(all_ys) + 0.3)
    ax3.set_xlabel("X (m)", weight='bold')
    ax3.set_ylabel("Y (Height m)", weight='bold')
    ax3.set_title("Front View (X-Y Plane)", weight='bold', fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # Subplot 4: Side View (Z-Y)
    ax4 = fig.add_subplot(2, 2, 4)
    for obj in objects:
        pts = obj["8points_rel"]
        poly_corners = pts[[0, 4, 6, 2], :][:, [2, 1]]
        polygon = patches.Polygon(poly_corners, closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.6, lw=1.2)
        ax4.add_patch(polygon)
        if obj["category"] != "floor":
            ax4.text(obj["z_rel"], obj["y_rel"] + obj["h"]/2, obj["category"], ha='center', va='center', fontsize=8, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax4.set_aspect('equal')
    ax4.set_xlim(min(all_zs) - margin, max(all_zs) + margin)
    ax4.set_ylim(min(all_ys) - 0.2, max(all_ys) + 0.3)
    ax4.set_xlabel("Z (Depth m)", weight='bold')
    ax4.set_ylabel("Y (Height m)", weight='bold')
    ax4.set_title("Side View (Z-Y Plane)", weight='bold', fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f"3D-FRONT Ground Truth Bounding Boxes: {target_scene}", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    bbox3d_local_path = os.path.join(debug_bbox_dir, f"{target_scene}_GT_3D_Bboxes.png")
    bbox3d_artifact_path = os.path.join(artifact_dir, f"{target_scene}_GT_3D_Bboxes.png")
    plt.savefig(bbox3d_local_path, bbox_inches='tight', dpi=200)
    plt.savefig(bbox3d_artifact_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved 3D Bounding Boxes multi-view plot to {bbox3d_local_path}")

    # =========================================================================
    # VISUALIZATION 3: Scene Graph Spatial Relationships
    # =========================================================================
    G = nx.DiGraph()
    node_labels = {}
    node_colors_list = []
    
    obj_dict = {obj["id"]: obj for obj in objects}
    for obj in objects:
        node_id = f"{obj['category']}_{obj['id']}"
        node_labels[node_id] = f"{obj['category']}\n(ID {obj['id']})"
        G.add_node(node_id)
        
    spatial_rels_filter = {'left', 'right', 'front', 'behind', 'above', 'close by', 'standing on'}
    
    for rel in scene_rel['relationships']:
        src_id, dest_id, _, rel_type = rel
        src_str, dest_str = str(src_id), str(dest_id)
        if rel_type in spatial_rels_filter and src_str in obj_dict and dest_str in obj_dict:
            # Skip standing on floor for all furniture to avoid cluttering floor node
            if rel_type == 'standing on' and obj_dict[dest_str]['category'] == 'floor':
                continue
            src_node = f"{obj_dict[src_str]['category']}_{src_str}"
            dest_node = f"{obj_dict[dest_str]['category']}_{dest_str}"
            G.add_edge(src_node, dest_node, label=rel_type)

    for node in G.nodes():
        obj_id = node.split('_')[-1]
        node_colors_list.append(obj_dict[obj_id]["color"])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.2)
    
    nx.draw_networkx_nodes(G, pos, node_size=3200, node_color=node_colors_list, edgecolors='black', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family="sans-serif", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.8, arrowstyle='-|>', arrowsize=20, edge_color='#555555', connectionstyle="arc3,rad=0.1", ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='#c0392b', font_weight='bold', ax=ax)
    
    ax.set_title(f"Scene Graph Spatial Relationships: {target_scene}", fontsize=13, weight='bold', pad=12)
    ax.axis('off')
    
    graph_local_path = os.path.join(debug_bbox_dir, f"{target_scene}_GT_SceneGraph.png")
    graph_artifact_path = os.path.join(artifact_dir, f"{target_scene}_GT_SceneGraph.png")
    plt.savefig(graph_local_path, bbox_inches='tight', dpi=200)
    plt.savefig(graph_artifact_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved Scene Graph to {graph_local_path}")

    # =========================================================================
    # VISUALIZATION 4: Interactive 3D HTML Bounding Box Visualization (Plotly)
    # =========================================================================
    fig_plotly = go.Figure()
    
    for obj in objects:
        pts = obj["8points_rel"]
        cat = obj["category"]
        color = obj["color"]
        
        # 8 corners
        x_pts, y_pts, z_pts = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Mesh3d faces
        # 12 triangles forming the 6 rectangular faces of the box
        i_faces = [0, 1, 0, 1, 0, 1, 4, 5, 0, 4, 1, 5]
        j_faces = [1, 5, 2, 3, 4, 6, 5, 7, 2, 6, 3, 7]
        k_faces = [5, 4, 3, 7, 6, 2, 7, 6, 6, 2, 7, 3]
        
        hover_text = (
            f"<b>{cat.upper()}</b> (ID: {obj['id']})<br>"
            f"Dimensions (L x H x W): {obj['l']:.2f}m x {obj['h']:.2f}m x {obj['w']:.2f}m<br>"
            f"Center (X, Y, Z): ({obj['x_rel']:.2f}, {obj['y_rel']:.2f}, {obj['z_rel']:.2f})<br>"
            f"Rotation Yaw: {obj['angle_deg']:.1f}°<br>"
            f"Model Path: {obj['model_path'] or 'N/A'}"
        )
        
        opacity = 0.2 if cat == "floor" else 0.55
        
        fig_plotly.add_trace(go.Mesh3d(
            x=x_pts, y=z_pts, z=y_pts, # Plotly Z is height, so swap Y and Z
            i=i_faces, j=j_faces, k=k_faces,
            color=color,
            opacity=opacity,
            name=f"{cat} (ID {obj['id']})",
            hoverinfo="text",
            text=hover_text,
            showscale=False
        ))
        
        # Wireframe lines for 12 edges
        for e1, e2 in edges:
            fig_plotly.add_trace(go.Scatter3d(
                x=[pts[e1, 0], pts[e2, 0]],
                y=[pts[e1, 2], pts[e2, 2]],
                z=[pts[e1, 1], pts[e2, 1]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False,
                hoverinfo='none'
            ))

    fig_plotly.update_layout(
        title=dict(
            text=f"Interactive 3D-FRONT Ground Truth Bounding Boxes: {target_scene}",
            font=dict(size=16, family="Arial", color="black")
        ),
        scene=dict(
            xaxis=dict(title="X (meters)", backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title="Z (Depth, meters)", backgroundcolor="rgb(240, 240, 240)"),
            zaxis=dict(title="Y (Height, meters)", backgroundcolor="rgb(240, 240, 240)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    html_local_path = os.path.join(debug_bbox_dir, f"{target_scene}_GT_3D_Interactive.html")
    html_artifact_path = os.path.join(artifact_dir, f"{target_scene}_GT_3D_Interactive.html")
    
    fig_plotly.write_html(html_local_path)
    fig_plotly.write_html(html_artifact_path)
    print(f"Saved Interactive 3D HTML visualization to {html_local_path}")

    print("\nALL GROUND TRUTH VISUALIZATIONS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
