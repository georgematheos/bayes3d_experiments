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

ObjectInfo = namedtuple("ObjectInfo", [
    "parent_obj", "parent_face", "child_face",
    "category_index", "root_pose", "params"
])
def empty_object_info():
    """
    Placeholder ObjectInfo for a nonexistant object.
    (This way we can store "no object" instances in the same
    memory layout of an ObjectInfo.)
    """
    return ObjectInfo(
        jnp.array(-1000),
        jnp.array(-1000),
        jnp.array(-1000),
        jnp.array(-1000),
        jnp.zeros((4, 4)),
        -10000 * jnp.ones(3)
    )
def is_valid(obj: ObjectInfo):
    """
    Check if an object is valid (i.e. exists).
    """
    return obj.category_index >= -1

@genjax.static_gen_fn
def generate_object(
    object_idx : int,
    possible_category_indices,
    pose_bounds,
    contact_bounds
):
    i = object_idx
    # parent_obj = uniform_discrete(-1, object_idx) @ f"parent_obj"
    min = jax.lax.select(i == 0, -1, 0) # only allow floating first object
    parent_obj = uniform_discrete(min, object_idx) @ f"parent_obj"
    parent_face = uniform_discrete(0, 6) @ f"face_parent"
    child_face = uniform_discrete(0, 6) @ f"face_child"
    category_index = uniform_choice(possible_category_indices) @ f"category_index"
    
    # TODO: use mask or switch so we only generate whichever of `pose` and `params`
    # is necessary for the current object.
    root_pose = (
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

    return ObjectInfo(parent_obj, parent_face, child_face, category_index, root_pose, params)

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
    "n_objects",
    "object_info", # Vectorized ObjectInfo - exactly the first n_objects are valid
    "poses"
])

def call_renderer(*args):
    # print("Calling renderer...")
    return b.RENDERER.render(*args)

@genjax.static_gen_fn
def model(
    max_n_objects_array, # array with shape[0] = max n objects
    possible_object_indices,
    pose_bounds,
    contact_bounds,
    all_box_dims
):
    max_n_objects = max_n_objects_array.shape[0]
    
    n_objects = uniform_discrete(0, max_n_objects) @ "n_objects"

    # This gives us a masked `object_info : ObjectInfo`
    # storing n real objects and (max_n_objects - n) empty objects.
    # It stores this by storing each value in `ObjectInfo` as
    # an array with leading dimension of size max_n_objects
    masked_object_info = generate_objects(
        jnp.arange(max_n_objects) < n_objects,
        (
            jnp.arange(max_n_objects),
            jnp.repeat(possible_object_indices[None, ...], max_n_objects, 0),
            jnp.repeat(pose_bounds[None, ...], max_n_objects, 0),
            jnp.repeat(contact_bounds[None, ...], max_n_objects, 0),
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
        is_valid(object_info)[:, None],
        all_box_dims[object_info.category_index],
        jnp.zeros(3)
    )
    poses = jnp.where(
        is_valid(object_info)[:, None, None],
        b.scene_graph.poses_from_scene_graph(
            object_info.root_pose,
            valid_box_dims,
            object_info.parent_obj, object_info.params,
            object_info.parent_face, object_info.child_face
        ),
        jnp.zeros((max_n_objects, 4, 4))
    )

    rendered = call_renderer(jnp.linalg.inv(camera_pose) @ poses, object_info.category_index)[..., :3]\
    
    variance = genjax.uniform(0.00000000001, 10000.0) @ "variance"
    outlier_prob = genjax.uniform(-0.01, 10000.0) @ "outlier_prob"
    noisy_image = image_likelihood(rendered, variance, outlier_prob) @ "image"

    return ModelOutput(rendered, n_objects, object_info, poses)

### Utils ###

def viz_trace_meshcat(trace, colors=None):
    out = trace.get_retval()
    b.clear_visualizer()
    b.show_cloud(
        "noisy_image",
        b.apply_transform_jit(trace["image"].reshape(-1, 3),
                              trace["camera_pose"]
                            )
    )
    b.show_cloud(
        "rendered_image",
        b.apply_transform_jit(
            out.rendered.reshape(-1, 3), trace["camera_pose"]
        ),
        color=b.RED,
    )
    indices = out.object_info.category_index
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(indices)))
    for i in range(out.n_objects):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[indices[i]], color=colors[i])
        b.set_pose(f"obj_{i}", out.poses[i])
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