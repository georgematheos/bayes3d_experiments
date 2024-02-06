import jax
import jax.numpy as jnp

def select_elements(cond, a, b):
    """
    Select elements from a or b based on the condition.

    Args:
    cond (array): A 1D array of boolean values with length equal to the size of the first dimension of a and b.
    a (array): An array of any shape, but its first dimension must be the same size as cond.
    b (array): An array of the same shape as a.

    Returns:
    array: An array of the same shape as a and b, with elements selected from a or b based on cond.
    """
    # Reshape cond to be broadcastable with a and b
    new_shape = (-1,) + (1,) * (a.ndim - 1)
    cond_broadcasted = cond.reshape(new_shape)

    # Use jnp.where to select elements
    return jnp.where(cond_broadcasted, a, b)

def select_or_default(cond, a, default):
    """
    Select elements from a or default based on the condition.
    
    Args:
    cond (array): A 1D array of boolean values with length equal to the size of the first dimension of a.
    a (array): An array of any shape, but its first dimension must be the same size as cond.
    default: A value of shape equal to the shape of a, minus the leading dimension.

    Returns:
    array: An array of the same shape as a, with elements selected from a or default based on cond.
    """
    b = jnp.broadcast_to(default, a.shape)
    return select_elements(cond, a, b)

def mymatch(masked, none, some):
    """
    A replacement for `genjax.Mask.match`, which looks to me like it may
    currently be buggy.
    """
    flag = jnp.array(masked.mask)
    sel_fn = lambda x, y : select_or_default(flag, x, y)
    return jax.tree_util.tree_map(sel_fn, some(masked.value), none())

