"""Tests for planning.data.ops."""

import collections

from planning.data import ops

import numpy as np

_TestNamedtuple = collections.namedtuple('_TestNamedtuple', ['test_field'])


def test_nested_map():
    inp = [(1, 2, 3), _TestNamedtuple(4)]
    out = ops.nested_map(lambda x: x + 1, inp)
    assert out == [(2, 3, 4), _TestNamedtuple(5)]


def test_nested_zip():
    inp = [_TestNamedtuple(1), _TestNamedtuple(2)]
    out = ops.nested_zip(inp)
    assert out == _TestNamedtuple([1, 2])


def test_nested_unzip():
    inp = _TestNamedtuple([1, 2])
    out = ops.nested_unzip(inp)
    assert out == [_TestNamedtuple(1), _TestNamedtuple(2)]


def test_nested_stack_unstack():
    inp = [_TestNamedtuple(1), _TestNamedtuple(2)]
    out = ops.nested_unstack(ops.nested_stack(inp))
    assert inp == out


def test_nested_unstack_stack():
    inp = _TestNamedtuple(np.array([1, 2]))
    out = ops.nested_unstack(ops.nested_stack(inp))
    np.testing.assert_equal(inp, out)


def test_nested_concatenate():
    inp = (
        _TestNamedtuple(np.array([1, 2])),
        _TestNamedtuple(np.array([3])),
    )
    out = ops.nested_concatenate(inp)
    np.testing.assert_equal(out, _TestNamedtuple(np.array([1, 2, 3])))
