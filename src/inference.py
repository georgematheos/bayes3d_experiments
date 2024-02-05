import inspect
from collections import namedtuple
import jax
import jax.numpy as jnp
import bayes3d as b
from .model import model
import genjax
from genjax import Diff, NoChange, UnknownChange

def add_object(trace, key, obj_id, parent, face_parent, face_child):
    r = trace.get_retval()
    N = r.indices.shape[0] + 1
    choices = genjax.choice_map({
        f"parent_{N-1}": parent,
        f"id_{N-1}": obj_id,
        f"face_parent_{N-1}": face_parent,
        f"face_child_{N-1}": face_child,
        f"contact_params_{N-1}": jnp.zeros(3)
    }).merge(trace.get_choices())[0]
    trace, weight = model.importance(key, choices, (jnp.arange(N), *trace.get_args()[1:]))
    return trace
add_object_jit = jax.jit(add_object)

def print_trace(trace):
    print(
        """
    SCORE: {:0.7f}
    VARIANCE: {:0.7f}
    OUTLIER_PROB {:0.7f}
    """.format(trace.get_score(), trace["variance"], trace["outlier_prob"])
    )
