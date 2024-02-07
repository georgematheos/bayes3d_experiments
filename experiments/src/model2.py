from collections import namedtuple
import genjax
import jax
import jax.numpy as jnp

import bayes3d as b

from .masking_utils import mymatch
from .genjax_distributions import (
    contact_params_uniform,
    image_likelihood,
    uniform_choice,
    uniform_discrete,
    uniform_pose,
)

ObjectInfo = namedtuple("ObjectInfo", ["parent_obj", "parent_face", "child_face", "category_index", "pose", "params"])
def empty_object_info():
    """
    Placeholder ObjectInfo for a nonexistant object.
    (This way we can store "no object" instances in the same
    memory layout of an ObjectInfo.)
    """
    return ObjectInfo(
        jnp.array(-1),
        jnp.array(-1),
        jnp.array(-1),
        jnp.array(-1),
        jnp.nan * jnp.zeros((4, 4)),
        jnp.nan * jnp.zeros(3)
    )
def is_valid(obj: ObjectInfo):
    """
    Check if an object is valid (i.e. exists).
    """
    return obj.category_index != -1

@genjax.static_gen_fn
def generate_object(
    object_idx : int,
    possible_category_indices,
    pose_bounds,
    contact_bounds
):
    i = object_idx
    parent_obj = uniform_discrete(-1, object_idx) @ f"parent_obj"
    parent_face = uniform_discrete(0, 6) @ f"face_parent"
    child_face = uniform_discrete(0, 6) @ f"face_child"
    category_index = uniform_choice(possible_category_indices) @ f"category_index"
    
    # TODO: use mask or switch so we only generate whichever of `pose` and `params`
    # is necessary for the current object.
    pose = (
            uniform_pose(
                pose_bounds[0],
                pose_bounds[1],
            )
            @ f"root_pose"
        )
    params = (
        contact_params_uniform(contact_bounds[0], contact_bounds[1])
        @ f"contact_params"
    )

    return ObjectInfo(parent_obj, parent_face, child_face, category_index, pose, params)

"""
Args:
    - Boolean array of length max_n_objects, for which objects to generate
    - Tuple of:
        - Array of object indices
        - Array of possible category indices
        - Array of pose bounds
        - Array of contact bounds
        Each array should have leading dimension equal to max_n_objects
"""
generate_objects = genjax.map_combinator(
    in_axes=(0, (0, 0, 0, 0))
)(genjax.masking_combinator(generate_object))

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
    max_n_objects : genjax.PytreeConst,
    n_objects : int,
    possible_object_indices,
    pose_bounds,
    contact_bounds,
    all_box_dims
):
    # This gives us a masked `object_info : ObjectInfo`
    # storing n real objects and (max_n_objects - n) empty objects.
    # It stores this by storing each value in `ObjectInfo` as
    # an array with leading dimension of size max_n_objects
    masked_object_info = generate_objects(
        jnp.arange(max_n_objects.const) < n_objects,
        (
            jnp.arange(max_n_objects.const),
            jnp.repeat(possible_object_indices[None, ...], max_n_objects.const, 0),
            jnp.repeat(pose_bounds[None, ...], max_n_objects.const, 0),
            jnp.repeat(contact_bounds[None, ...], max_n_objects.const, 0),
        )
    ) @ "objects"
    object_info = mymatch(
        masked_object_info,
        lambda: empty_object_info(),
        lambda x: x
    )
    
    camera_pose = uniform_pose(
        pose_bounds[0],
        pose_bounds[1],
    ) @ "camera_pose"

    valid_box_dims = jnp.where(
        (object_info.category_index == -1)[:, None],
        jnp.zeros(3),
        all_box_dims[object_info.category_index]
    )
    poses = b.scene_graph.poses_from_scene_graph(
        object_info.pose,
        valid_box_dims,
        object_info.parent_obj, object_info.params,
        object_info.parent_face, object_info.child_face
    )

    rendered = b.RENDERER.render(jnp.linalg.inv(camera_pose) @ poses, object_info.category_index)[..., :3]
    print("Got rendered.")
    print(rendered)

    variance = genjax.uniform(0.00000000001, 10000.0) @ "variance"
    outlier_prob = genjax.uniform(-0.01, 10000.0) @ "outlier_prob"
    noisy_image = image_likelihood(rendered, variance, outlier_prob) @ "image"
    print("Got noisy image.")

    return ModelOutput(
        rendered,
        object_info.category_index,
        poses,
        object_info.parent_obj,
        object_info.params,
        object_info.parent_face,
        object_info.child_face,
        object_info.pose,
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

### Scratch work ###

# mymatch(object_info,
#     lambda: jnp.nan * jnp.ones((4, 4)),
#     lambda x : b.scene_graph.poses_from_scene_graph(
#         x.pose, all_box_dims[x.category_index], x.parent_obj,
#         x.params, x.parent_face, x.child_face
#     )               
# )

# poses = b.scene_graph.poses_from_scene_graph(
#     mymatch(object_info, lambda: jnp.zeros((4, 4)), lambda x: x.pose),
#     mymatch(object_info, lambda: jnp.zeros((3,)), lambda x: all_box_dims[x.category_index]),
#     mymatch(object_info, lambda: jnp.array(-1), lambda x: x.parent_obj),
#     mymatch(object_info, lambda: jnp.array(-1), lambda x: x.params),
#     mymatch(object_info, lambda: jnp.array(-1), lambda x: x.parent_face),
#     mymatch(object_info, lambda: jnp.array(-1), lambda x: x.child_face)
# )