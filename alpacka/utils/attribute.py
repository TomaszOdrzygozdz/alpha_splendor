"""Handling attributes helpers."""

import functools


def rgetattr(obj, path, *default):
    """Recursive getattr."""
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def rsetattr(obj, path, value):
    """Recursive setattr."""
    pre, _, post = path.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj,
                   post,
                   value)
