import os
import json
import numpy as np
import trimesh
import glob
import cv2
import torch
import seaborn as sns
import argparse

if "PYOPENGL_PLATFORM" in os.environ:
    del os.environ["PYOPENGL_PLATFORM"]

def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot

def fit_shapes_to_box_v2(obj, box, degrees=False):
    l, h, w, px, py, pz, angle = box
    if isinstance(l, torch.Tensor):
        l, h, w, px, py, pz, angle = l.item(), h.item(), w.item(), px.item(), py.item(), pz.item(), angle.item()
    box_points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                box_points.append([l / 2 * i, h * j, w / 2 * k])

    bounding_box = obj.bounding_box
    bottom_center = bounding_box.bounds[0] + (bounding_box.extents / 2)
    bottom_center[1] = bounding_box.bounds[0][1]
    rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])
    translation_matrix = trimesh.transformations.translation_matrix(-bottom_center)
    transform = np.dot(translation_matrix, rotation_matrix)
    obj.apply_transform(transform)

    R = get_rotation_3dfront(angle, degree=degrees)
    R_inv = np.linalg.inv(R)
    t = np.array([px, py, pz])
    T = np.concatenate((R_inv,t.reshape(-1,1)),axis=1)
    T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
    vertices = np.array(obj.vertices)
    shape_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    obj.apply_scale(1 / shape_size)
    obj.apply_scale([l, h, w])
    obj.apply_transform(T)
    box_points = np.asarray(box_points)
    box_points = box_points.dot(R)
    box_points += np.expand_dims(t, 0)
    return box_points, obj

def create_bbox_edges(box, color, thickness=0.03):
    l, h, w, px, py, pz, angle = box
    if isinstance(l, torch.Tensor):
        l, h, w, px, py, pz, angle = l.item(), h.item(), w.item(), px.item(), py.item(), pz.item(), angle.item()
    
    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                points.append([l/2 * i, h * j, w/2 * k])
    points = np.array(points)
    
    R = get_rotation_3dfront(angle, degree=True)
    t = np.array([px, py, pz])
    points = points.dot(R) + t
    
    edges = [
        (0,1), (2,3), (4,5), (6,7), # Z edges
        (0,2), (1,3), (4,6), (5,7), # Y edges
        (0,4), (1,5), (2,6), (3,7)  # X edges
    ]
    
    edge_meshes = []
    for (idx1, idx2) in edges:
        p1 = points[idx1]
        p2 = points[idx2]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-4: continue
        
        cyl = trimesh.creation.cylinder(radius=thickness, height=length)
        z_axis = np.array([0, 0, 1.0])
        vec_norm = vec / length
        axis = np.cross(z_axis, vec_norm)
        angle_rot = np.arccos(np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0))
        
        if np.linalg.norm(axis) < 1e-6:
            if vec_norm[2] < 0:
                T_rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            else:
                T_rot = np.eye(4)
        else:
            T_rot = trimesh.transformations.rotation_matrix(angle_rot, axis)
            
        T_trans = trimesh.transformations.translation_matrix((p1 + p2) / 2)
        cyl.apply_transform(np.dot(T_trans, T_rot))
        
        cyl.visual.vertex_colors = color
        cyl.visual.face_colors = color
        edge_meshes.append(cyl)
        
    return edge_meshes

def render_img(trimesh_meshes):
    import pyrender
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    for tri_mesh in trimesh_meshes:
        if len(tri_mesh.vertices) == 0: continue
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)
    camera_location = np.array([0.0, 8.0, 0.0])  # y axis
    look_at_point = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, -1.0])  # -z axis

    camera_direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    right_vector = np.cross(camera_direction, up_vector)
    up_vector = np.cross(right_vector, camera_direction)

    camera_pose = np.identity(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = up_vector
    camera_pose[:3, 2] = -camera_direction
    camera_pose[:3, 3] = camera_location
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=camera_pose)
    
    color, depth = renderer.render(scene)
    return color

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True, help="E.g., LivingRoom-1097")
    parser.add_argument("--target_folder", required=True, help="E.g., baseline or complete_released_full_model")
    parser.add_argument("--json_name", default="physcene_collision_resolved.json", help="Which json file to use")
    parser.add_argument("--old_mesh_dir", default="/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/echoscene/object_meshes")
    parser.add_argument("--mode", choices=['bbox_only', 'both'], default='bbox_only', help="Whether to output only the bounding boxes, or both objects and bounding boxes.")
    args = parser.parse_args()

    base_path = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene"
    target_dir = os.path.join(base_path, args.target_folder, "vis/2050")
    
    in_json = os.path.join(target_dir, args.json_name)
    if not os.path.exists(in_json):
        # Fallback to another possible name if resolved doesn't exist
        in_json = os.path.join(target_dir, "physcene_collision_input_all.json")
        if not os.path.exists(in_json):
            in_json = os.path.join(target_dir, "physcene_collision_input.json")
    
    if not os.path.exists(in_json):
        print(f"Error: Could not find JSON file in {target_dir}")
        return

    out_scene_dir = os.path.join(target_dir, "render_bboxes_glb")
    os.makedirs(out_scene_dir, exist_ok=True)

    print(f"Loading JSON from {in_json}...")
    with open(in_json, 'r') as f:
        data = json.load(f)

    if args.scene_id not in data["scene_ids"]:
        print(f"Error: Scene {args.scene_id} not found in JSON.")
        return
        
    scene_idx = data["scene_ids"].index(args.scene_id)
    
    class_labels = np.array(data["class_labels"][scene_idx])
    cats = np.argmax(class_labels, axis=1)
    translations = np.array(data["translations"][scene_idx])
    sizes = np.array(data["sizes"][scene_idx])
    angles_rad = np.array(data["angles"][scene_idx])
    angles_deg = angles_rad * 180.0 / np.pi
    
    boxes = np.concatenate([sizes, translations], axis=-1)
    box_and_angle = np.concatenate([boxes, angles_deg], axis=-1)
    box_and_angle = torch.tensor(box_and_angle).float()

    num_classes = class_labels.shape[1] - 1
    color_palette = np.array(sns.color_palette('hls', num_classes)) * 255.0

    old_mesh_scene_dir = os.path.join(args.old_mesh_dir, args.scene_id)

    trimesh_meshes = []
    classes = [str(c) for c in range(num_classes)]
    
    instance_id = 1
    for j in range(len(cats)):
        cat_id = cats[j]
        is_object = data["objectness"][scene_idx][j][0] == 1.0
        
        if not is_object:
            if cat_id < num_classes:
                classes[cat_id] = '_scene_'
            continue
            
        pattern = os.path.join(old_mesh_scene_dir, f"*_{cat_id}_{instance_id}.obj")
        matched_files = glob.glob(pattern)
        
        if len(matched_files) == 0:
            print(f"  Warning: No matching obj found for {pattern}")
            instance_id += 1
            continue
        
        filepath = matched_files[0]
        query_label = os.path.basename(filepath).split('_')[0]
        if cat_id < num_classes:
            classes[cat_id] = query_label
        
        # Load mesh
        obj = trimesh.load(filepath, force='mesh')
        
        # Make the mesh mostly transparent or gray? 
        # The user said "render the glb with the 3d bbox with the edge color matching the object color"
        # Let's keep the mesh color as is but maybe slightly dim it, or just use the object color.
        # Original pipeline colors the mesh with the category color.
        color = [int(c) for c in color_palette[cat_id]] + [255]
        
        # Apply color to mesh
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        
        # Fit shape
        box_pts, obj_fitted = fit_shapes_to_box_v2(obj, torch.tensor(box_and_angle[j]).float(), degrees=True)
        
        if args.mode == 'both':
            trimesh_meshes.append(obj_fitted)
        
        # Create 3D bbox edges with the same color
        bbox_edges = create_bbox_edges(box_and_angle[j], color, thickness=0.03)
        trimesh_meshes.extend(bbox_edges)
            
        instance_id += 1

    print(f"Exporting {args.scene_id} with {len(trimesh_meshes)} components (meshes + bbox edges)...")
    try:
        scene = trimesh.Scene(trimesh_meshes)
        out_path = os.path.join(out_scene_dir, f"{args.scene_id}_bbox.glb")
        scene.export(out_path)
        print(f"Saved GLB to {out_path}")
    except Exception as e:
        print(f"Error exporting GLB: {e}")

if __name__ == '__main__':
    main()
