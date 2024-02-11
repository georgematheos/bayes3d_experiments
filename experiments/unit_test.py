"""
This is a unit test that confirms inference in a synthetic scene
rendered from PyBullet identifies the correct objects.
"""

import os
import glob
import pickle
import jax
import jax.numpy as jnp
import bayes3d as b
import genjax
import matplotlib.pyplot as plt

from src.model import model

# TODO for those extending this unit test:
# We really ought to run this test N times,
# with a different key each time.
key = jax.random.PRNGKey(0)

### Load data & setup renderer ###
NEAR = 0.1
FAR = 5.0

with open('../data.pkl', 'rb') as f:
    data = pickle.load(f)

camera_image_1 = data["init"][0]
categories_on_table = data["init"][1]
target_category = data["init"][2]
table_info = data["init"][3] # table pose and dimensions
n_objects = 5

X_WT = b.t3d.pybullet_pose_to_transform(table_info[0])
X_WC = b.t3d.pybullet_pose_to_transform(camera_image_1["camera_pose"])
X_CT = b.t3d.inverse_pose(X_WC) @ X_WT

def image_to_rgbd(camera_image_1):
    K = camera_image_1['camera_matrix']
    rgb = camera_image_1['rgbPixels']
    depth = camera_image_1['depthPixels']
    camera_pose = camera_image_1['camera_pose']
    camera_pose = b.t3d.pybullet_pose_to_transform(camera_pose)
    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = depth.shape
    rgbd_original = b.RGBD(rgb, depth, camera_pose, b.Intrinsics(h,w,fx,fy,cx,cy,NEAR,FAR))
    return rgbd_original

rgbd_original = image_to_rgbd(camera_image_1)
print("Got rgbd_original")

scaling_factor = 0.2
rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd_original, scaling_factor)
print("Got rgb immage scaled down.")

b.setup_renderer(rgbd_scaled_down.intrinsics)

model_dir = os.path.join(os.path.abspath('../..'), 'bayes3d/assets/bop/ycbv/models')
mesh_path = os.path.join(model_dir,"obj_" + "{}".format(13+1).rjust(6, '0') + ".ply")
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

### Model + inference ###
    
def flat_choicemaps_to_vector_choicemap(choicemaps):
    cm = genjax.choice_map({
        k : jnp.array([c.get_submap(k).get_value() for c in choicemaps])
        for (k, _) in choicemaps[0].get_submaps_shallow()
    })
    return cm

obs_img = b.unproject_depth_jit(
    rgbd_scaled_down.depth,
    rgbd_scaled_down.intrinsics
)

table_choicemap = genjax.choice_map({
                    "root_pose": X_CT,
                    "category_index": 0,
                    "parent_obj": -1,
                    "face_parent": 2,
                    "face_child": 3,
                    "contact_params": jnp.zeros(3)
                })
def obj_choicemap(category_idx):
    return genjax.choice_map({
        "root_pose": jnp.eye(4),
        "category_index": category_idx,
        "parent_obj": 0,
        "face_parent": 2,
        "face_child": 3,
        "contact_params": jnp.zeros(3)
    })

max_n_objects = 8
# Initialize the trace with the table, and all
# the other objects initialized to be on the table
# with the identity pose.
map_choicemap = genjax.indexed_choice_map(
            jnp.arange(1),
            flat_choicemaps_to_vector_choicemap([
                table_choicemap
            ])
        )
constraints = genjax.choice_map({
    "n_objects": 1,
    "camera_pose": jnp.eye(4),
    "image": b.unproject_depth_jit(
        rgbd_scaled_down.depth,
        rgbd_scaled_down.intrinsics
    ),
    "objects": map_choicemap,
    "variance": 0.02,
    "outlier_prob": 0.0005
})

model_args = (
    jnp.arange(max_n_objects), # max_n_objects_array
    # possible_object_indices
    jnp.arange(len(b.RENDERER.meshes)),
    # pose_bounds
    jnp.array([-jnp.ones(3)*5.0, jnp.ones(3)*5.0]),
    # contact_bounds
    jnp.array([jnp.array([-1., -1., -jnp.pi]), jnp.array([1., 1., jnp.pi])]),
    # all_box_dims
    b.RENDERER.model_box_dims
)

subkey, key = jax.random.split(key)
tr1, w1 = model.importance(subkey, constraints, model_args)

def add_object_choicemap(prev_n_objects, cat_idx):
    obj_idx = prev_n_objects
    return genjax.choice_map({
        "n_objects": prev_n_objects + 1,
        "objects": genjax.indexed_choice_map(
            jnp.array([obj_idx]),
            flat_choicemaps_to_vector_choicemap([obj_choicemap(cat_idx)])
        )
    })

def add_object(tr, cat_idx):
    n_objects = tr["n_objects"]
    newtr, _, _, _ = tr.update(
        key,
        add_object_choicemap(n_objects, cat_idx)
    )
    return newtr

grid_params = [
    (0.65, jnp.pi, (30,30,15)),
    (0.2, jnp.pi, (15,15,15)),
    (0.1, jnp.pi, (15,15,15)),
    (0.05, jnp.pi, (15,15,15)),
    (0.02, jnp.pi, (9,9,51)),
    (0.01, jnp.pi/5, (15, 15, 15))
]

contact_param_gridding_schedule = [
    b.utils.make_translation_grid_enumeration_3d(
        -x, -x, -ang,
        x, x, ang,
        *nums
    )
    for (x,ang,nums) in grid_params
]

def cp_choicemap(object_idx, v):
    return genjax.choice_map({
            "objects": genjax.indexed_choice_map(
                jnp.array([object_idx]),
                flat_choicemaps_to_vector_choicemap([genjax.choice_map({"contact_params": v})])
            )
        })
def _c2f(key, tr, object_idx, contact_param_gridding_schedule):
    updater = jax.jit(jax.vmap(lambda trace, v: trace.update(
        key, cp_choicemap(object_idx, v)
    )[0].get_score(), in_axes=(None, 0)))
    cp = tr.get_retval().object_info.params[object_idx, ...]
    for cp_grid in contact_param_gridding_schedule:
        cps = cp + cp_grid
        scores = updater(tr, cps)
        cp = cps[jnp.argmax(scores)]
    potential_trace = tr.update(key, cp_choicemap(object_idx, cp))[0]
    return potential_trace

c2f = jax.jit(_c2f)

def _extend_then_c2f(key, tr, cat_idx, contact_param_gridding_schedule):
    return c2f(
        key,
        add_object(tr, cat_idx),
        tr["n_objects"],
        contact_param_gridding_schedule
    )
extend_then_c2f = jax.jit(_extend_then_c2f)

def extend_c2f_all_categories(key, tr, contact_param_gridding_schedule):
    indices = jnp.arange(len(b.RENDERER.meshes))
    keys = jax.random.split(key, len(indices))
    potential_trs = [extend_then_c2f(key, tr, idx, contact_param_gridding_schedule) for (key, idx) in zip(keys, indices)]
    scores = jnp.array([t.get_score() for t in potential_trs])
    best_idx = jnp.argmax(scores)
    return potential_trs[best_idx]

def extend_c2f_all_categories_n_times(key, tr, contact_param_gridding_schedule, n):
    for _ in range(n):
        tr = extend_c2f_all_categories(key, tr, contact_param_gridding_schedule)
    return tr

def fit_objects_until_convergence(key, tr, contact_param_gridding_schedule, eps=0.1, max_iter=10):
    old_score = -jnp.inf
    new_score = tr.project(genjax.select("image"))
    i = 1
    print("Adding first object...")
    while new_score > old_score + eps and i < max_iter:
        newtr = extend_c2f_all_categories(key, tr, contact_param_gridding_schedule)
        old_score = new_score
        new_score = newtr.project(genjax.select("image"))
        if new_score > old_score + eps:
            tr = newtr
        print(f"Added object {i}.  Image score = {new_score}.")
        i += 1
    return tr

### Unit test ###

inferred_trace = fit_objects_until_convergence(key, tr1, contact_param_gridding_schedule)

# Test that inference worked
assert inferred_trace["n_objects"] == 7
cats = inferred_trace.get_retval().object_info.category_index
for (count, idx) in [
    (1, 0), (1, 4), (1, 2), (1, 3), (2, 1), (1, 5)
]:
    assert jnp.count_nonzero(cats == idx) == count

print("All tests passed")
