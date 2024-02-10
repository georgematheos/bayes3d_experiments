from typing import NamedTuple
import genjax
import jax
import jax.numpy as jnp

import bayes3d

from .masking_utils import mymatch
from .genjax_distributions import (
    contact_params_uniform,
    image_likelihood,
    uniform_choice,
    uniform_discrete,
    uniform_pose,
)

class ObjectInfo(NamedTuple):
    parent_obj: int # Index of parent object in scene graph
                    # Is -1 if there is no parent object.
    parent_face: int # Face of parent object that this object is attached to
                     # 3 = top face
    child_face: int # Face of this object that is attached to parent
                    # 2 = bottom face
    category_index: int # Index of mesh in Bayes3D renderer for this object
    root_pose: jnp.ndarray # Object pose, if parent_object = -1
                           # (4, 4) pose matrix
    params: jnp.ndarray # (x, y, theta) coordinates of this object
                        # in the parent object's frame, if parent_object != -1
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
    possible_category_indices, # Array of shape (n_categories, )
    pose_bounds, # Array of shape (3, 2): [[minx, miny, minz], [maxx, maxy, maxz]]
    contact_bounds # Array of shape (3, 2): [[minx, miny, mintheta], [maxx, maxy, maxtheta]]
):
    min = jax.lax.select(object_idx == 0, -1, 0)
    parent_obj = uniform_discrete(min, object_idx) @ f"parent_obj"  
    category_index = uniform_choice(possible_category_indices) @ f"category_index"
    parent_face = uniform_discrete(0, 6) @ f"face_parent"
    child_face = uniform_discrete(0, 6) @ f"face_child"
    
    # If parent_obj = -1, this value will still be generated, but then ignored.
    root_pose = (
            uniform_pose(
                pose_bounds[0],
                pose_bounds[1],
            )
            @ f"root_pose"
        )
    
    # If parent_obj = -1, this value will still be generated, but then ignored.
    contact_params = (
        contact_params_uniform(contact_bounds[0], contact_bounds[1])
        @ f"contact_params"
    )

    return ObjectInfo(parent_obj, parent_face, child_face, category_index, root_pose, contact_params)
    
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
    in_axes=(0, (0, None, None, None))
)(genjax.masking_combinator(generate_object))


class ModelOutput(NamedTuple):
    rendered: jnp.ndarray
    n_objects: int
    object_info: ObjectInfo # Vectorized ObjectInfo - exactly the first n_objects are valid
    poses: jnp.ndarray

@genjax.static_gen_fn
def model(
    max_n_objects_array, # array with shape = (max n objects,)
    possible_object_indices, # Array, shape = (n_objects,)
    pose_bounds, # Array of shape (3, 2): [[minx, miny, minz], [maxx, maxy, maxz]]
    contact_bounds, # Array of shape (3, 2): [[minx, miny, mintheta], [maxx, maxy, maxtheta]]
    
    # Last arg: bounding box dimensions for the object mesh for each category,
    # [xsize, ysize, zsize], in meters.
    all_box_dims # Array of shape (n_categories, 3)
):
    ### Generate the objects in the scene ###

    # Choose a random number of objects to generate
    max_n_objects = max_n_objects_array.shape[0]
    n_objects = uniform_discrete(0, max_n_objects) @ "n_objects"

    masked_object_info = generate_objects(
        jnp.arange(max_n_objects) < n_objects, # is-active array
        (
            jnp.arange(max_n_objects), # object_index
            possible_object_indices,
            pose_bounds,
            contact_bounds
        )
    ) @ "objects"

    # Get the underlying ObjectInfo objects, or empty ones if they don't exist.
    object_info = mymatch(
        masked_object_info,
        lambda: empty_object_info(),
        lambda x: x
    )

    ### Compute the 6DOF object poses from the scene graph generated above ###
    
    # Bounding boxes dimensions for each object mesh; [0, 0, 0] for each null object slot
    valid_box_dims = jnp.where(
        is_valid(object_info)[:, None],
        all_box_dims[object_info.category_index],
        jnp.zeros(3)
    )
    # Compute the 6DOF poses of each object in the scene using
    # bayes3d.scene_graph.poses_from_scene_graph;
    # if the object is not valid, we just return jnp.zeros((4, 4)) to fill
    # a pose-shaped memory slot with null values.
    poses = jnp.where(
        is_valid(object_info)[:, None, None],
        bayes3d.scene_graph.poses_from_scene_graph(
            object_info.root_pose,
            valid_box_dims,
            object_info.parent_obj, object_info.params,
            object_info.parent_face, object_info.child_face
        ),
        jnp.zeros((max_n_objects, 4, 4))
    )

    ### Generate a camera pose and generate a picture of the scene ###
    camera_pose = uniform_pose(
        pose_bounds[0],
        pose_bounds[1],
    ) @ "camera_pose"
    rendered = bayes3d.RENDERER.render(jnp.linalg.inv(camera_pose) @ poses, object_info.category_index)[..., :3]\
    
    # Prior on noise parameters for the noisy image likelihood
    variance = genjax.uniform(0.00000000001, 10000.0) @ "variance"

    # Generate a noisy image of the scene
    noisy_image = image_likelihood(rendered, variance, 0.) @ "image"

    return ModelOutput(rendered, n_objects, object_info, poses)

### Utils ###

def viz_trace_meshcat(trace, colors=None):
    out = trace.get_retval()
    bayes3d.clear_visualizer()
    bayes3d.show_cloud(
        "noisy_image",
        bayes3d.apply_transform_jit(trace["image"].reshape(-1, 3),
                              trace["camera_pose"]
                            )
    )
    bayes3d.show_cloud(
        "rendered_image",
        bayes3d.apply_transform_jit(
            out.rendered.reshape(-1, 3), trace["camera_pose"]
        ),
        color=bayes3d.RED,
    )
    indices = out.object_info.category_index
    if colors is None:
        colors = bayes3d.viz.distinct_colors(max(10, len(indices)))
    for i in range(out.n_objects):
        bayes3d.show_trimesh(f"obj_{i}", bayes3d.RENDERER.meshes[indices[i]], color=colors[i])
        bayes3d.set_pose(f"obj_{i}", out.poses[i])
    bayes3d.show_pose("camera_pose", trace["camera_pose"])

def viz_trace_rendered_observed(trace, scale=2):
    return bayes3d.viz.hstack_images(
        [
            bayes3d.viz.scale_image(
                bayes3d.get_depth_image(trace.get_retval().rendered[..., 2]), scale
            ),
            bayes3d.viz.scale_image(bayes3d.get_depth_image(trace["image"][..., 2]), scale),
        ]
    )