"""Ops for manipulating pytrees.

Pytrees are nested structures of lists/tuples/namedtuples. They typically have
numpy arrays at leaves.

Behavior of those ops is well described in planning.data.ops_test.
"""

import numpy as np


def _is_leaf(x):
    """Returns whether pytree is a leaf."""
    return not isinstance(x, (tuple, list))


def _is_namedtuple_instance(x):
    """Determines if x is an instance of a namedtuple type."""
    if isinstance(x, tuple):
        return hasattr(x, '_fields')
    else:
        return False


def nested_map(f, x, stop_fn=_is_leaf):
    """Maps a function through a pytree.

    Args:
        f: (callable) Function to map.
        x: Pytree to map over.
        stop_fn: (callable) Optional stopping condition for the recursion. By
            default, stops on leaves.
    """
    if stop_fn(x):
        return f(x)

    if _is_namedtuple_instance(x):
        return type(x)(*nested_map(f, tuple(x), stop_fn=stop_fn))
    assert isinstance(x, (list, tuple)), (
        'Non-exhaustive pattern match for {}.'.format(type(x))
    )
    return type(x)(nested_map(f, y, stop_fn=stop_fn) for y in x)


def nested_zip(xs):
    """Zips a list of pytrees.

    Inverse of nested_unzip.

    Example:
        nested_zip([((1, 2), 3), ((4, 5), 6)]) == (([1, 4], [2, 5]), [3, 6])
    """
    assert not _is_leaf(xs)
    assert xs
    for x in xs:
        assert type(x) is type(xs[0]), (  # noqa: E721, check exact types
            'Cannot zip pytrees of different types: '
            '{} and {}.'.format(type(x), type(xs[0]))
        )

    if _is_namedtuple_instance(x):
        return type(xs[0])(*nested_zip([tuple(x) for x in xs]))
    elif isinstance(xs[0], (list, tuple)):
        for x in xs:
            assert len(x) == len(xs[0]), (
                'Cannot zip sequences of different lengths: '
                '{} and {}'.format(len(x), len(xs[0]))
            )

        return type(xs[0])(
            nested_zip([x[i] for x in xs]) for i in range(len(xs[0]))
        )
    else:
        return xs


def _is_last_level(x):
    """Returns whether pytree is at the last level (children are leaves)."""
    return not _is_leaf(x) and all(map(_is_leaf, x))


def nested_unzip(x):
    """Uzips a pytree of lists.

    Inverse of nested_unzip.

    Example:
        nested_unzip((([1, 4], [2, 5]), [3, 6])) == [((1, 2), 3), ((4, 5), 6)]
    """
    acc = []
    try:
        i = 0
        while True:
            acc.append(nested_map(
                lambda l: l[i],
                x,
                stop_fn=_is_last_level,
            ))
            i += 1
    except IndexError:
        return acc


def nested_stack(xs):
    """Stacks a list of pytrees of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_stack([(1, 2), (3, 4)]) == (np.array([1, 3]), np.array([2, 4]))
    """
    return nested_map(np.stack, nested_zip(xs), stop_fn=_is_last_level)


def nested_unstack(x):
    """Unstacks a pytree of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_unstack((np.array([1, 3]), np.array([2, 4]))) == [(1, 2), (3, 4)]
    """
    def unstack(arr):
        (*slices,) = arr
        return slices
    return nested_unzip(nested_map(unstack, x))


def nested_concatenate(xs):
    """Concatenates a list of pytrees of numpy arrays."""
    return nested_map(np.concatenate, nested_zip(xs), stop_fn=_is_last_level)