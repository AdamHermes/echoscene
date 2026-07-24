import os
import glob
import shutil
import trimesh
import cv2
import numpy as np
import pyrender
import json
import torch
import seaborn as sns

# Disable EGL backend for macOS compatibility
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
    return obj

def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
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

def process_folder(json_path, raw_mesh_dir, out_dir):
    print(f"Processing to out folder: {out_dir}")
        
    out_mesh_dir_base = os.path.join(out_dir, "vis/2050/echoscene/object_meshes")
    out_scene_dir = os.path.join(out_dir, "vis/2050/echoscene")
    out_img_dir = os.path.join(out_dir, "vis/2050/render_imgs/echoscene")
    
    os.makedirs(out_mesh_dir_base, exist_ok=True)
    os.makedirs(out_scene_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Copy JSON over
    out_json_path = os.path.join(out_dir, "vis/2050", os.path.basename(json_path))
    with open(out_json_path, 'w') as f:
        json.dump(data, f)
        
    num_scenes = len(data["scene_ids"])
    for i in range(num_scenes):
        scene_id = data["scene_ids"][i]
        print(f"[{i+1}/{num_scenes}] Processing {scene_id}...")
        
        class_labels = np.array(data["class_labels"][i])
        cats = np.argmax(class_labels, axis=1)
        translations = np.array(data["translations"][i])
        sizes = np.array(data["sizes"][i])
        angles_rad = np.array(data["angles"][i])
        angles_deg = angles_rad * 180.0 / np.pi
        
        boxes = np.concatenate([sizes, translations], axis=-1)
        box_and_angle = np.concatenate([boxes, angles_deg], axis=-1)
        
        num_classes = class_labels.shape[1] - 1
        color_palette = np.array(sns.color_palette('hls', num_classes))
        
        scene_mesh_dir = os.path.join(out_mesh_dir_base, scene_id)
        os.makedirs(scene_mesh_dir, exist_ok=True)
        
        old_mesh_scene_dir = os.path.join(raw_mesh_dir, scene_id)
        
        trimesh_meshes = []
        instance_id = 1
        for j in range(len(cats)):
            cat_id = cats[j]
            is_object = data["objectness"][i][j][0] == 1.0
            
            if not is_object:
                continue
                
            pattern = os.path.join(old_mesh_scene_dir, f"*_{cat_id}_{instance_id}.obj")
            matched_files = glob.glob(pattern)
            
            if len(matched_files) == 0:
                instance_id += 1
                continue
                
            filepath = matched_files[0]
            query_label = os.path.basename(filepath).split('_')[0]
            
            # Skip lamps!
            if query_label == 'lamp' or cat_id == 7:
                instance_id += 1
                continue
                
            obj = trimesh.load(filepath, force='mesh')
            color = color_palette[cat_id]
            obj.visual.vertex_colors = color
            obj.visual.face_colors = color
            
            obj.export(os.path.join(scene_mesh_dir, os.path.basename(filepath)))
            
            obj_fitted = fit_shapes_to_box_v2(obj, torch.tensor(box_and_angle[j]).float(), degrees=True)
            trimesh_meshes.append(obj_fitted)
                
            instance_id += 1
            
        if len(trimesh_meshes) > 0:
            scene = trimesh.Scene(trimesh_meshes)
            scene.export(os.path.join(out_scene_dir, f"{scene_id}_echoscene.glb"))
            
            try:
                color_img = render_img(trimesh_meshes)
                color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(os.path.join(out_img_dir, f"{scene_id}.png"), color_bgr)
            except Exception as e:
                print(f"Error rendering {scene_id}: {e}")

def main():
    root_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene"
    
    # We always pull the raw, unshifted unscaled .obj files from the original baseline folder!
    raw_mesh_dir = os.path.join(root_dir, "baseline/vis/2050/echoscene/object_meshes")
    
    # 1. Original (Before Resolution) without lamp
    orig_json = os.path.join(root_dir, "baseline/vis/2050/physcene_collision_input.json")
    orig_out = os.path.join(root_dir, "baseline_without_lamp")
    
    # 2. Post Processed (After Resolution) without lamp
    post_json = os.path.join(root_dir, "baseline_post_processed/vis/2050/physcene_collision_input.json")
    post_out = os.path.join(root_dir, "baseline_post_processed_without_lamp")
    
    # First, let's just make sure the destination folders are clean
    if os.path.exists(orig_out):
        shutil.rmtree(orig_out)
    if os.path.exists(post_out):
        shutil.rmtree(post_out)
        
    process_folder(orig_json, raw_mesh_dir, orig_out)
    process_folder(post_json, raw_mesh_dir, post_out)
    
    print("Finished processing all folders without lamps. Renders are now correctly scaled!")

if __name__ == "__main__":
    main()
