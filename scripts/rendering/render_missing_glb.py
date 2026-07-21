import os
import glob
import trimesh
import cv2
import numpy as np

# Use default OpenGL backend for mac compatibility
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for tri_mesh in trimesh_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)

    # set up positions and the origin, exactly as in visualize_scene.py
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
    glb_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/echoscene"
    out_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/render_imgs/echoscene"
    
    os.makedirs(out_dir, exist_ok=True)
    
    glb_files = glob.glob(os.path.join(glb_dir, "*.glb"))
    print(f"Found {len(glb_files)} .glb files.")
    
    rendered = 0
    skipped = 0
    
    for glb_file in glb_files:
        filename = os.path.basename(glb_file)
        scene_name = filename.replace('_echoscene.glb', '')
        out_path = os.path.join(out_dir, f"{scene_name}.png")
        
        if os.path.exists(out_path):
            skipped += 1
            continue
            
        print(f"Rendering {scene_name}...")
        try:
            scene = trimesh.load(glb_file, force='scene')
            trimesh_meshes = list(scene.geometry.values())
                
            color_img = render_img(trimesh_meshes)
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(out_path, color_bgr)
            rendered += 1
        except Exception as e:
            print(f"Failed to render {glb_file}: {e}")
            
    print(f"Finished! Rendered {rendered} new images. Skipped {skipped} existing images.")

if __name__ == "__main__":
    main()
