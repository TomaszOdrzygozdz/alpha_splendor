"""Tests for planning.data.ops."""

from planning.data import messages
from planning.data import ops

import numpy as np


def test_nested_map():
    inp = [(1, 2, 3), messages.PredictRequest(4)]
    out = ops.nested_map(lambda x: x + 1, inp)
    assert out == [(2, 3, 4), messages.PredictRequest(5)]


def test_nested_zip():
    inp = [messages.PredictRequest(1), messages.PredictRequest(2)]
    out = ops.nested_zip(inp)
    assert out == messages.PredictRequest([1, 2])


def test_nested_unzip():
    inp = messages.PredictRequest([1, 2])
    out = ops.nested_unzip(inp)
    assert out == [messages.PredictRequest(1), messages.PredictRequest(2)]


def test_nested_stack_unstack():
    inp = [messages.PredictRequest(1), messages.PredictRequest(2)]
    out = ops.nested_unstack(ops.nested_stack(inp))
    assert inp == out


def test_nested_unstack_stack():
    inp = messages.PredictRequest(np.array([1, 2]))
    out = ops.nested_unstack(ops.nested_stack(inp))
    np.testing.assert_equal(inp, out)


def test_nested_concatenate():
    inp = (
        messages.PredictRequest(np.array([1, 2])),
        messages.PredictRequest(np.array([3])),
    )
    out = ops.nested_concatenate(inp)
    np.testing.assert_equal(out, messages.PredictRequest(np.array([1, 2, 3])))
