"""Utilities for gym spaces."""

import gym

from alpacka import data


def space_iter(action_space):
    """Returns an iterator over points in a gym space."""
    try:
        return iter(action_space)
    except TypeError:
        if isinstance(action_space, gym.spaces.Discrete):
            return iter(range(action_space.n))
        else:
            raise TypeError('Space {} does not support iteration.'.format(
                type(action_space)
            ))


def space_signature(space):
    """Returns a TensorSignature of elements of the given space."""
    return data.TensorSignature(shape=space.shape, dtype=space.dtype)
