import pickle
import os
import glob
import bayes3d as b
import jax.numpy as jnp

def load_nice_intrinsics():
    NEAR = 0.1
    FAR = 5.0

    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    camera_image_1 = data["init"][0]
    K = camera_image_1['camera_matrix']
    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = camera_image_1['depthPixels'].shape
    return b.Intrinsics(h,w,fx,fy,cx,cy,NEAR,FAR)

def get_nice_table_pose():
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    camera_image_1 = data["init"][0]
    table_info = data["init"][3] # table pose and dimensions
    X_WT = b.t3d.pybullet_pose_to_transform(table_info[0])
    X_WC = b.t3d.pybullet_pose_to_transform(camera_image_1["camera_pose"])
    X_CT = b.t3d.inverse_pose(X_WC) @ X_WT
    return X_CT

def image_to_rgbd(camera_image_1):
    K = camera_image_1['camera_matrix']
    rgb = camera_image_1['rgbPixels']
    depth = camera_image_1['depthPixels']
    camera_pose = camera_image_1['camera_pose']
    camera_pose = b.t3d.pybullet_pose_to_transform(camera_pose)
    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = depth.shape                                                          # NEAR,  FAR
    rgbd_original = b.RGBD(rgb, depth, camera_pose, b.Intrinsics(h,w,fx,fy,cx,cy, 0.1, 5.0))
    return rgbd_original

def load_pybullet_obs_img():
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    camera_image_1 = data["init"][0]

    rgbd_original = image_to_rgbd(camera_image_1)
    print("Got rgbd_original")

    scaling_factor = 0.2
    rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd_original, scaling_factor)
    print("Got rgb immage scaled down.")

    return (rgbd_original, rgbd_scaled_down)

def load_some_object_meshes(override=False):
    if len(b.RENDERER.meshes) > 0 and not override:
        print("Meshes already loaded.  Use override=True to add more meshes anyway.")
        return

    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    categories_on_table = data["init"][1]
    table_info = data["init"][3] # table pose and dimensions

    model_dir = os.path.join(os.path.abspath('../..'), 'bayes3d/assets/bop/ycbv/models')
    ycb_filenames = glob.glob(os.path.join(model_dir, "*.ply"))
    ycb_index_order = [int(s.split("/")[-1].split("_")[-1].split(".")[0]) for s in ycb_filenames]
    sorted_ycb_filenames = [s for _,s in sorted(zip(ycb_index_order, ycb_filenames))]

    relevant_objects = [any(x in name for x in categories_on_table) for (i, name) in enumerate(b.utils.ycb_loader.MODEL_NAMES)]
    relevant_object_names = [b.utils.ycb_loader.MODEL_NAMES[i] for i in range(len(b.utils.ycb_loader.MODEL_NAMES)) if relevant_objects[i]]
    filtered_filenames = [sorted_ycb_filenames[i] for i in range(len(sorted_ycb_filenames)) if relevant_objects[i]]

    table_dims = table_info[1:]
    table_mesh = b.utils.make_cuboid_mesh(table_dims)
    b.RENDERER.add_mesh(table_mesh, "table")
    print("Added table mesh.")

    pillar_mesh = b.utils.make_cuboid_mesh(jnp.array([0.02, 0.02, 0.5]))
    b.RENDERER.add_mesh(pillar_mesh, "pillar")
    print("Added pillar mesh.")

    for model_path in filtered_filenames:
        b.RENDERER.add_mesh_from_file(model_path, scaling_factor=1.0/1000.0)
        print(f"Added mesh at path {model_path}.")

    return ["table", "pillar", *relevant_object_names]