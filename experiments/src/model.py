from collections import namedtuple
import genjax
import jax
import jax.numpy as jnp
# from genjax.incremental import Diff, NoChange, UnknownChange

import bayes3d as b

from .genjax_distributions import (
    contact_params_uniform,
    image_likelihood,
    uniform_choice,
    uniform_pose,
)

ModelOutput = namedtuple("ModelOutput", [
        "rendered",
        "indices",
        "poses",
        "parents",
        "contact_params",
        "faces_parents",
        "faces_child",
        "root_poses",
    ])

@genjax.static_gen_fn
def model(
    n_obj_array,             # Array of length = num objects in the scene
    possible_object_indices, # B3D renderer indices for possible object categories
    pose_bounds,             # (2, 3) array of min and max pose bounds (X, Y, Z space)
    contact_bounds,          # (2, 3) array of min and max contact bounds (in X, Y, theta space)
    all_box_dims             # List of (3,) arrays: [length, width, height] for each object mesh BBox
):
    indices = jnp.array([], dtype=jnp.int32)
    root_poses = jnp.zeros((0, 4, 4))
    contact_params = jnp.zeros((0, 3))
    faces_parents = jnp.array([], dtype=jnp.int32)
    faces_child = jnp.array([], dtype=jnp.int32)
    parents = jnp.array([], dtype=jnp.int32)
    for i in range(n_obj_array.shape[0]):
        parent_obj = (
            uniform_choice(jnp.arange(-1, n_obj_array.shape[0] - 1)) @ f"parent_{i}"
        )
        parent_face = uniform_choice(jnp.arange(0, 6)) @ f"face_parent_{i}"
        child_face = uniform_choice(jnp.arange(0, 6)) @ f"face_child_{i}"
        index = uniform_choice(possible_object_indices) @ f"id_{i}"

        pose = (
            uniform_pose(
                pose_bounds[0],
                pose_bounds[1],
            )
            @ f"root_pose_{i}"
        )

        params = (
            contact_params_uniform(contact_bounds[0], contact_bounds[1])
            @ f"contact_params_{i}"
        )

        indices = jnp.concatenate([indices, jnp.array([index])])
        root_poses = jnp.concatenate([root_poses, pose.reshape(1, 4, 4)])
        contact_params = jnp.concatenate([contact_params, params.reshape(1, -1)])
        parents = jnp.concatenate([parents, jnp.array([parent_obj])])
        faces_parents = jnp.concatenate([faces_parents, jnp.array([parent_face])])
        faces_child = jnp.concatenate([faces_child, jnp.array([child_face])])

    box_dims = all_box_dims[indices]
    poses = b.scene_graph.poses_from_scene_graph(
        root_poses, box_dims, parents, contact_params, faces_parents, faces_child
    )

    camera_pose = (
        uniform_pose(
            pose_bounds[0],
            pose_bounds[1],
        )
        @ "camera_pose"
    )

    rendered = b.RENDERER.render(jnp.linalg.inv(camera_pose) @ poses, indices)[..., :3]

    variance = genjax.uniform(0.00000000001, 10000.0) @ "variance"
    outlier_prob = genjax.uniform(-0.01, 10000.0) @ "outlier_prob"
    _image = image_likelihood(rendered, variance, outlier_prob) @ "image"

    return ModelOutput(
        rendered,
        indices,
        poses,
        parents,
        contact_params,
        faces_parents,
        faces_child,
        root_poses,
    )

### Utils ###

def viz_trace_meshcat(trace, colors=None):
    b.clear_visualizer()
    b.show_cloud(
        "1", b.apply_transform_jit(trace["image"].reshape(-1, 3), trace["camera_pose"])
    )
    b.show_cloud(
        "2",
        b.apply_transform_jit(
            trace.get_retval().rendered.reshape(-1, 3), trace["camera_pose"]
        ),
        color=b.RED,
    )
    indices = trace.get_retval().indices
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(indices)))
    for i in range(len(indices)):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[indices[i]], color=colors[i])
        b.set_pose(f"obj_{i}", trace.get_retval().poses[i])
    b.show_pose("camera_pose", trace["camera_pose"])

def viz_trace_rendered_observed(trace, scale=2):
    return b.viz.hstack_images(
        [
            b.viz.scale_image(
                b.get_depth_image(trace.get_retval().rendered[..., 2]), scale
            ),
            b.viz.scale_image(b.get_depth_image(trace["image"][..., 2]), scale),
        ]
    )
