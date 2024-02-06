from dataclasses import dataclass

import jax
import jax.numpy as jnp
from genjax.core.datatypes import JAXGenerativeFunction
from genjax.generative_functions.distributions import ExactDensity

import bayes3d as b

class GaussianVMFPose(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf(key, pose_mean, var, concentration)

    def logpdf(self, pose, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf_logpdf(pose, pose_mean, var, concentration)


class UniformPose(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, low, high, **kwargs):
        position = jax.random.uniform(key, shape=(3,)) * (high - low) + low
        orientation = b.quaternion_to_rotation_matrix(
            jax.random.normal(key, shape=(4,))
        )
        return b.transform_from_rot_and_pos(orientation, position)

    def logpdf(self, pose, low, high, **kwargs):
        position = pose[:3, 3]
        valid = (low <= position) & (position <= high)
        position_score = jnp.log(
            (valid * 1.0) * (jnp.ones_like(position) / (high - low))
        )
        return position_score.sum() + jnp.pi**2


class ImageLikelihood(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, img, variance, outlier_prob):
        return img

    def logpdf(self, observed_image, latent_image, variance, outlier_prob):
        return b.threedp3_likelihood(
            observed_image,
            latent_image,
            variance,
            outlier_prob,
        )


class ContactParamsUniform(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, low, high):
        return jax.random.uniform(key, shape=(3,)) * (high - low) + low

    def logpdf(self, sampled_val, low, high, **kwargs):
        valid = (low <= sampled_val) & (sampled_val <= high)
        log_probs = jnp.log((valid * 1.0) * (jnp.ones_like(sampled_val) / (high - low)))
        return log_probs.sum()


class UniformDiscreteArray(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, vals, arr):
        return jax.random.choice(key, vals, shape=arr.shape)

    def logpdf(self, sampled_val, vals, arr, **kwargs):
        return jnp.log(1.0 / (vals.shape[0])) * arr.shape[0]

class UniformChoice(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, vals):
        return jax.random.choice(key, vals)

    def logpdf(self, sampled_val, vals, **kwargs):
        valid = jnp.isin(sampled_val, vals)
        log_probs = jnp.where(valid, -jnp.log(vals.shape[0]), -jnp.inf)
        return log_probs

class UniformDiscrete(ExactDensity, JAXGenerativeFunction):
    """
    uniform_discrete(a, b) samples a uniform integer x such that a <= x < b.
    If a is not less than b, the result is always a.
    """
    def sample(self, key, low, high):
        return jax.random.randint(key, shape=(), minval=low, maxval=high)

    def logpdf(self, sampled_val, low, high, **kwargs):
        range_is_nontrivial = low + 1 <= high
        equals_low = low == sampled_val
    
        is_in_range = (low <= sampled_val) & (sampled_val < high)

        log_probs_branch1 = jnp.where(equals_low, 0., -jnp.inf)
        log_probs_branch2 = jnp.where(is_in_range, -jnp.log(high - low), -jnp.inf)
        log_probs = jnp.where(range_is_nontrivial, log_probs_branch2, log_probs_branch1)

        return log_probs

gaussian_vmf_pose = GaussianVMFPose()
image_likelihood = ImageLikelihood()
contact_params_uniform = ContactParamsUniform()
uniform_discrete = UniformDiscrete()
uniform_choice = UniformChoice()
uniform_discrete_array = UniformDiscreteArray()
uniform_pose = UniformPose()
