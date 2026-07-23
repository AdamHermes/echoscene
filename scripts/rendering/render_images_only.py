import os
import json
import numpy as np
import trimesh
import glob
import cv2
import sys
import torch
import seaborn as sns
from pathlib import Path

# Remove PYOPENGL_PLATFORM to prevent mac crash
if "PYOPENGL_PLATFORM" in os.environ:
    del os.environ["PYOPENGL_PLATFORM"]
    
def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot

def params_to_8points_3dfront(box, degrees=False):
    l, h, w, px, py, pz, angle = box
    if isinstance(l, torch.Tensor):
        l, h, w, px, py, pz, angle = l.item(), h.item(), w.item(), px.item(), py.item(), pz.item(), angle.item()
    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                points.append([l/2 * i, h * j, w/2 * k])
    points = np.asarray(points)
    points = points.dot(get_rotation_3dfront(angle, degree=degrees))
    points += np.expand_dims(np.array([px, py, pz]), 0)
    return points

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

def create_bg(box_and_angle, cat_ids, classes, type='floor'):
    points_list_x = []
    points_list_y = []
    points_list_z = []
    for j in range(0, box_and_angle.shape[0]):
        if cat_ids[j] >= len(classes):
            continue
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_':
            continue
        box_points = params_to_8points_3dfront(box_and_angle[j], degrees=True)
        points_list_x.append(box_points[0:2, 0])
        points_list_x.append(box_points[6:8, 0])
        points_list_y.append(box_points[0:2, 1])
        points_list_y.append(box_points[6:8, 1])
        points_list_z.append(box_points[0:2, 2])
        points_list_z.append(box_points[6:8, 2])

    if len(points_list_x) == 0:
        return trimesh.Trimesh()

    points_x = np.array(points_list_x).reshape(-1,1)
    points_y = np.array(points_list_y).reshape(-1,1)
    points_z = np.array(points_list_z).reshape(-1,1)
    points = np.concatenate((points_x,points_y, points_z),axis=1)
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    if type == 'floor':
        vertices = np.array([[min_x, min_y, min_z],
                             [min_x, min_y, max_z],
                             [max_x, min_y, max_z],
                             [max_x, min_y, min_z]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
    elif type == 'walls':
        vertices1 = np.array([[min_x, min_y, min_z],
                             [min_x, min_y, max_z],
                             [min_x, max_y, max_z],
                             [min_x, max_y, min_z]], dtype=np.float32) # min x
        faces1 = np.array([[1, 0, 3], [1, 3, 2]])
        vertices2 = np.array([[max_x, min_y, min_z],
                                   [min_x, min_y, min_z],
                                   [min_x, max_y, min_z],
                                   [max_x, max_y, min_z]], dtype=np.float32) # min z
        faces2 = np.array([[1, 0, 3], [1, 3, 2]])
        vertices3 = np.array([[max_x, min_y, min_z],
                                   [max_x, min_y, max_z],
                                   [max_x, max_y, max_z],
                                   [max_x, max_y, min_z]], dtype=np.float32) # max x
        faces3 = np.array([[0, 1, 2], [0, 2, 3]])
        vertices4 = np.array([[min_x, min_y, max_z],
                                   [max_x, min_y, max_z],
                                   [max_x, max_y, max_z],
                                   [min_x, max_y, max_z]], dtype=np.float32) # max z
        faces4 = np.array([[1, 0, 3], [1, 3, 2]])
        vertices = np.concatenate([vertices1, vertices2, vertices3, vertices4])
        faces = np.concatenate([faces1, faces2 + len(vertices1), faces3 + len(vertices1) + len(vertices2),
                                faces4 + len(vertices1) + len(vertices2) + len(vertices3)])
    elif type == 'ceiling':
        vertices = np.array([[min_x, max_y, min_z],
                             [min_x, max_y, max_z],
                             [max_x, max_y, max_z],
                             [max_x, max_y, min_z]], dtype=np.float32)
        faces = np.array([[1, 0, 3], [1, 3, 2]])
    else:
        raise NotImplementedError
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def render_img(trimesh_meshes):
    import pyrender
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", default="/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json")
    parser.add_argument("--old_mesh_dir", default="/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/echoscene/object_meshes")
    parser.add_argument("--out_base_dir", default="/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model_post_processed/vis/2050")
    args = parser.parse_args()

    in_json = args.in_json
    old_mesh_dir_base = args.old_mesh_dir
    out_base_dir = args.out_base_dir

    out_img_dir = os.path.join(out_base_dir, "render_imgs")
    os.makedirs(out_img_dir, exist_ok=True)

    print("Loading JSON...")
    with open(in_json, 'r') as f:
        data = json.load(f)

    num_scenes = len(data["scene_ids"])
    print(f"Processing {num_scenes} scenes...")

    for i in range(num_scenes):
        scene_id = data["scene_ids"][i]
        print(f"Processing scene: {scene_id} ({i+1}/{num_scenes})")
        
        class_labels = np.array(data["class_labels"][i]) # (N, num_classes+1)
        cats = np.argmax(class_labels, axis=1) # (N,)
        translations = np.array(data["translations"][i])
        sizes = np.array(data["sizes"][i])
        angles_rad = np.array(data["angles"][i])
        angles_deg = angles_rad * 180.0 / np.pi
        
        boxes = np.concatenate([sizes, translations], axis=-1)
        box_and_angle = np.concatenate([boxes, angles_deg], axis=-1)
        box_and_angle = torch.tensor(box_and_angle).float()

        num_classes = class_labels.shape[1] - 1
        color_palette = np.array(sns.color_palette('hls', num_classes))

        # skipping mesh dir creation

        old_mesh_scene_dir = os.path.join(old_mesh_dir_base, scene_id)

        trimesh_meshes = []
        lamp_mesh_list = []
        
        classes = [str(c) for c in range(num_classes)]
        
        instance_id = 1
        for j in range(len(cats)):
            cat_id = cats[j]
            is_object = data["objectness"][i][j][0] == 1.0
            
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
            
            obj = trimesh.load(filepath, force='mesh')
            color = color_palette[cat_id]
            obj.visual.vertex_colors = color
            obj.visual.face_colors = color
            
            # skip exporting obj
            
            box_points, obj_fitted = fit_shapes_to_box_v2(obj, torch.tensor(box_and_angle[j]).float(), degrees=True)
            if query_label == 'lamp':
                lamp_mesh_list.append(obj_fitted)
            else:
                trimesh_meshes.append(obj_fitted)
                
            instance_id += 1

        all_meshes = list(trimesh_meshes) + list(lamp_mesh_list)
        # In the original pipeline, floor/walls/ceiling are only appended if demo=True.
        # Since they lack materials, appending them makes them appear as black boxes in the .glb.
        # We omit them here to perfectly mirror the original dataset's outputs.
            
        # skip exporting glb
        
        try:
            color_img = render_img(all_meshes)
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(os.path.join(out_img_dir, f"{scene_id}.png"), color_bgr)
        except Exception as e:
            print(f"  Error rendering image for {scene_id}: {e}")

if __name__ == '__main__':
    main()
