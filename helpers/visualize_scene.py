import os

import open3d as o3d
import numpy as np
import trimesh

from helpers.util import fit_shapes_to_box, params_to_8points, params_to_8points_no_rot, params_to_8points_3dfront, get_database_objects, get_sdfusion_models, get_bbox, get_generated_shapes, trimeshes_to_pytorch3d, normalize_py3d_meshes
import json
import torch
import cv2
import pyrender
import seaborn as sns
from collections import OrderedDict
from model.diff_utils.util import tensor2im
from model.diff_utils.util_3d import render_sdf, render_meshes, sdf_to_mesh
from PIL import Image

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def tnsrs2ims(tensors):
    ims = []
    for tensor in tensors:
            ims.append(tensor2im(tensor))
    return ims

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def store_current_imgs(visuals, img_dir, classes, cat_ids):
    # write images to disk

    cat_ids = iter(cat_ids)
    for instance_id, image_numpy in visuals.items():
        cat_id = next(cat_ids)
        query_label = classes[cat_id].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            cat_id = next(cat_ids)
            query_label = classes[cat_id].strip('\n')
        img_path = os.path.join(img_dir, query_label+'_'+str(cat_id)+'_'+str(instance_id)+'.png')
        save_image(image_numpy, img_path)

def get_current_visuals_v2(vocab, renderer, meshes, obj_ids):
    texts = [vocab.get(id) for id in obj_ids]
    with torch.no_grad():
        py3d_meshes = trimeshes_to_pytorch3d(meshes)
        py3d_meshes = normalize_py3d_meshes(py3d_meshes)
        img_gen_df = render_meshes(renderer, py3d_meshes).detach().cpu()  # rendered generated sdf

    b, c, h, w = img_gen_df.shape
    img_shape = (3, h, w)

    vis_ims = tnsrs2ims(img_gen_df)
    visuals = zip(range(1,len(vis_ims)+1), vis_ims)

    return OrderedDict(visuals)

def create_bg(box_and_angle, cat_ids, classes, type='floor'):
    points_list_x = []
    points_list_y = []
    points_list_z = []
    for j in range(0, box_and_angle.shape[0]):
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

    points_x = np.array(points_list_x).reshape(-1,1)
    points_y = np.array(points_list_y).reshape(-1,1)
    points_z = np.array(points_list_z).reshape(-1,1)
    points = np.concatenate((points_x,points_y, points_z),axis=1)
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    from scipy.spatial import ConvexHull
    # Compute the 2D convex hull of all object points on the floor plane (X, Z)
    pts_2d = points[:, [0, 2]]
    hull = ConvexHull(pts_2d)
    
    # Extract the hull vertices in counter-clockwise order
    hull_pts = points[hull.vertices]
    
    if type == 'floor':
        vertices = np.zeros((len(hull_pts), 3), dtype=np.float32)
        vertices[:, 0] = hull_pts[:, 0]
        vertices[:, 1] = min_y
        vertices[:, 2] = hull_pts[:, 2]
        
        # Triangulate convex polygon (fan triangulation from vertex 0)
        faces = []
        for i in range(1, len(vertices) - 1):
            faces.append([0, i, i + 1])
        faces = np.array(faces)
        
    elif type == 'ceiling':
        vertices = np.zeros((len(hull_pts), 3), dtype=np.float32)
        vertices[:, 0] = hull_pts[:, 0]
        vertices[:, 1] = max_y
        vertices[:, 2] = hull_pts[:, 2]
        
        # Triangulate (reverse order for normal pointing down)
        faces = []
        for i in range(1, len(vertices) - 1):
            faces.append([0, i + 1, i])
        faces = np.array(faces)
        
    elif type == 'walls':
        vertices_list = []
        faces_list = []
        
        n = len(hull_pts)
        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            
            # 4 vertices for this wall segment
            v0 = [p1[0], min_y, p1[2]]
            v1 = [p1[0], max_y, p1[2]]
            v2 = [p2[0], max_y, p2[2]]
            v3 = [p2[0], min_y, p2[2]]
            
            idx = len(vertices_list)
            vertices_list.extend([v0, v1, v2, v3])
            
            # triangles
            faces_list.append([idx, idx + 2, idx + 1])
            faces_list.append([idx, idx + 3, idx + 2])
            
        vertices = np.array(vertices_list, dtype=np.float32)
        faces = np.array(faces_list)
    else:
        raise NotImplementedError
    return trimesh.Trimesh(vertices=vertices, faces=faces)



def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for tri_mesh in trimesh_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)

    # set up positions and the origin
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


def render_box(scene_id, cats, predBoxes, predAngles, datasize='small', classes=None, render_type='txt2shape',
               render_shapes=True, store_img=False, render_boxes=False, demo=False, visual=False, without_lamp=False,
               str_append="", mani=0, missing_nodes=None, manipulated_nodes=None, objs_before=None, store_path=None):
    os.makedirs(store_path,exist_ok=True)
    if render_type not in ['txt2shape', 'retrieval', 'onlybox']:
        raise ValueError('Render type needs to be either set to txt2shape or retrieval or onlybox.')
    color_palette = np.array(sns.color_palette('hls', len(classes)))
    box_and_angle = torch.cat([predBoxes.float(), predAngles.float()], dim=-1)

    obj_n = len(box_and_angle)
    if mani == 2:
        if len(missing_nodes) > 0:
            box_and_angle = box_and_angle[missing_nodes]
        elif len(manipulated_nodes) > 0:
            box_and_angle = box_and_angle[sorted(manipulated_nodes)]

    mesh_dir = os.path.join(store_path, render_type, 'object_meshes', scene_id[0])
    os.makedirs(mesh_dir, exist_ok=True)
    if render_type == 'retrieval':
        lamp_mesh_list, trimesh_meshes, raw_meshes = get_database_objects(box_and_angle, datasize, cats, classes, mesh_dir, render_boxes=render_boxes,
                                                 colors=color_palette[cats], without_lamp=without_lamp)
    elif render_type == 'txt2shape':
        lamp_mesh_list, trimesh_meshes, raw_meshes = get_sdfusion_models(box_and_angle, cats, classes, mesh_dir, render_boxes=render_boxes,
                                             colors=color_palette[cats], without_lamp=without_lamp)
    elif render_type == 'onlybox':
        lamp_mesh_list, trimesh_meshes = get_bbox(box_and_angle, cats, classes, colors=color_palette[cats], without_lamp=without_lamp)
    else:
        raise NotImplementedError

    if mani == 2:
        print("manipulated nodes: ", len(manipulated_nodes), len(trimesh_meshes))
        if len(missing_nodes) > 0:
            trimesh_meshes += objs_before
            query_label = classes[cats[0]].strip('\n')
            str_append += "_" + query_label
        elif len(manipulated_nodes) > 0:
            i, j, k = 0, 0, 0
            for i in range(obj_n):
                query_label = classes[cats[i]].strip('\n')
                i += 1
                if query_label == '_scene_' or query_label == 'floor' or (query_label == 'lamp' and without_lamp):
                    continue
                if i in manipulated_nodes:
                    objs_before[j] = trimesh_meshes[k]
                    str_append += "_" + query_label
                    #all_meshes.append(trimesh_meshes[j])
                    k += 1
                j += 1
            trimesh_meshes = objs_before

    if demo:
        mesh_dir_shifted = mesh_dir.replace('object_meshes', 'object_meshes_shifted')
        os.makedirs(mesh_dir_shifted, exist_ok=True)
        trimesh_meshes += lamp_mesh_list
        floor_mesh = create_bg(box_and_angle, cats, classes, type='floor')
        trimesh_meshes.append(floor_mesh)
        ceiling_mesh = create_bg(box_and_angle, cats, classes, type='ceiling')
        trimesh_meshes.append(ceiling_mesh)
        walls_mesh = create_bg(box_and_angle, cats, classes, type='walls')
        trimesh_meshes.append(walls_mesh)
        for i, mesh in enumerate(trimesh_meshes):
            mesh.export(os.path.join(mesh_dir_shifted,  f"{i}.obj"))
    scene = trimesh.Scene(trimesh_meshes)
    scene_path = os.path.join(store_path, render_type)
    if len(str_append) > 0:
        render_type += str_append
    os.makedirs(scene_path, exist_ok=True)
    scene.export(os.path.join(scene_path, "{0}_{1}.glb".format(scene_id[0], render_type)))

    if visual:
        scene.show()

    if store_img and not demo:
        img_path = os.path.join(store_path, render_type, "render_imgs")
        os.makedirs(img_path, exist_ok=True)
        color_img = render_img(trimesh_meshes)
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
        file_name = scene_id[0]
        if len(str_append) > 0:
            file_name += str_append
        cv2.imwrite(os.path.join(img_path, f'{file_name}.png'), color_bgr)

    if mani==1:
        return trimesh_meshes

def render_full(scene_id, cats, predBoxes, predAngles=None, datasize='small', classes=None, shapes_pred=None, render_type='txt2shape',
           render_shapes=True, store_img=False, render_boxes=False, demo=False, visual=False, epoch=None, no_stool = False, without_lamp=False, str_append="", mani=0, missing_nodes=None, manipulated_nodes=None, objs_before=None, store_path=None, save_3d = True):
    os.makedirs(store_path,exist_ok=True)

    if render_type not in ['echoscene', 'txt2shape', 'retrieval', 'onlybox']:
        raise ValueError('Render type needs to be either set to echoscene or txt2shape or retrieval or onlybox.')
    color_palette = np.array(sns.color_palette('hls', len(classes)))
    box_and_angle = torch.cat([predBoxes.float(), predAngles.float()], dim=1)

    obj_n = len(box_and_angle)
    if mani == 2:
        if len(missing_nodes) > 0:
            box_and_angle = box_and_angle[missing_nodes]
        elif len(manipulated_nodes) > 0:
            box_and_angle = box_and_angle[sorted(manipulated_nodes)]
    if save_3d:
        mesh_dir = os.path.join(store_path, render_type, 'object_meshes', scene_id[0])
        os.makedirs(mesh_dir, exist_ok=True)
    else:
        mesh_dir = None
    if render_type == 'echoscene':
        lamp_mesh_list, trimesh_meshes, raw_meshes = get_generated_shapes(box_and_angle, shapes_pred, cats, classes, mesh_dir, render_boxes=render_boxes, colors=color_palette[cats], without_lamp=without_lamp)

    elif render_type == 'retrieval':
        lamp_mesh_list, trimesh_meshes, raw_meshes = get_database_objects(box_and_angle, datasize, cats, classes, mesh_dir, render_boxes=render_boxes, colors=color_palette[cats], without_lamp=without_lamp)

    elif render_type == 'txt2shape':
        lamp_mesh_list, trimesh_meshes, raw_meshes = get_sdfusion_models(box_and_angle, cats, classes, mesh_dir, render_boxes=render_boxes, colors=color_palette[cats], no_stool=no_stool, without_lamp=without_lamp)

    elif render_type == 'onlybox':
        lamp_mesh_list, trimesh_meshes = get_bbox(box_and_angle, cats, classes, colors=color_palette[cats], without_lamp=without_lamp)
    else:
        raise NotImplementedError

    if mani == 2:
        print("manipulated nodes: ", len(manipulated_nodes), len(trimesh_meshes))
        if len(missing_nodes) >0:
            trimesh_meshes += objs_before
            query_label = classes[cats[0]].strip('\n')
            str_append += "_" + query_label
        elif len(manipulated_nodes) > 0:
            i, j, k = 0, 0, 0
            for i in range(obj_n):
                query_label = classes[cats[i]].strip('\n')
                i += 1
                if query_label == '_scene_' or query_label == 'floor' or (query_label == 'lamp' and without_lamp):
                    continue
                if i in manipulated_nodes:
                    objs_before[j] = trimesh_meshes[k]
                    str_append += "_" + query_label
                    #all_meshes.append(trimesh_meshes[j])
                    k += 1
                j += 1
            trimesh_meshes = objs_before

    if demo and save_3d:
        mesh_dir_shifted = mesh_dir.replace('object_meshes', 'object_meshes_shifted')
        os.makedirs(mesh_dir_shifted, exist_ok=True)
        trimesh_meshes += lamp_mesh_list
        floor_mesh = create_bg(box_and_angle, cats, classes, type='floor')
        trimesh_meshes.append(floor_mesh)
        ceiling_mesh = create_bg(box_and_angle, cats, classes, type='ceiling')
        trimesh_meshes.append(ceiling_mesh)
        walls_mesh = create_bg(box_and_angle, cats, classes, type='walls')
        trimesh_meshes.append(walls_mesh)
        for i, mesh in enumerate(trimesh_meshes):
            mesh.export(os.path.join(mesh_dir_shifted, f"{i}.obj"))
    scene = trimesh.Scene(trimesh_meshes)
    if len(str_append) > 0:
        # render_type_ = render_type + str_append
        render_type += str_append
    if save_3d:
        scene_path = os.path.join(store_path, render_type)
        os.makedirs(scene_path, exist_ok=True)
        scene.export(os.path.join(scene_path, "{0}_{1}.glb".format(scene_id[0], render_type)))

    if visual:
        scene.show()

    if store_img and not demo:
        img_path = os.path.join(store_path, "render_imgs", render_type)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        color_img = render_img(trimesh_meshes)
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
        file_name = scene_id[0]
        if len(str_append) > 0:
            file_name += str_append
        cv2.imwrite(os.path.join(img_path, '{}.png'.format(file_name)), color_bgr)