"""Handling attributes helpers."""

import functools


def recursive_getattr(obj, path, *default):
    """Recursive getattr."""
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def recursive_setattr(obj, path, value):
    """Recursive setattr."""
    pre, _, post = path.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj,
                   post,
                   value)
