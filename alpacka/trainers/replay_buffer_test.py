"""Tests for alpacka.trainers.replay_buffer."""

import collections

import numpy as np
import pytest

from alpacka.trainers import replay_buffer


_TestTransition = collections.namedtuple('_TestTransition', ['test_field'])

# Keep _TestTransitions with a single number in the buffer.
_test_datapoint_spec = _TestTransition(test_field=())


def test_samples_added_transition():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=10)
    stacked_transitions = _TestTransition(np.array([123]))
    buf.add(stacked_transitions)
    assert buf.sample(batch_size=1) == stacked_transitions


def test_raises_when_sampling_from_an_empty_buffer():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=10)
    with pytest.raises(ValueError):
        buf.sample(batch_size=1)


def test_samples_all_transitions_eventually_one_add():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=10)
    buf.add(_TestTransition(np.array([0, 1])))
    sampled_transitions = set()
    for _ in range(100):
        sampled_transitions.add(buf.sample(batch_size=1).test_field.item())
    assert sampled_transitions == {0, 1}


def test_samples_all_transitions_eventually_two_adds():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=10)
    buf.add(_TestTransition(np.array([0, 1])))
    buf.add(_TestTransition(np.array([2, 3])))
    sampled_transitions = set()
    for _ in range(100):
        sampled_transitions.add(buf.sample(batch_size=1).test_field.item())
    assert sampled_transitions == {0, 1, 2, 3}


def test_samples_different_transitions():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=100)
    buf.add(_TestTransition(np.arange(100)))
    assert len(set(buf.sample(batch_size=3).test_field)) > 1


def test_oversamples_transitions():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=10)
    stacked_transitions = _TestTransition(np.array([0, 1]))
    buf.add(stacked_transitions)
    assert set(buf.sample(batch_size=100).test_field) == {0, 1}


def test_overwrites_old_transitions():
    buf = replay_buffer.ReplayBuffer(_test_datapoint_spec, capacity=4)
    buf.add(_TestTransition(np.arange(3)))
    buf.add(_TestTransition(np.arange(3, 6)))
    # 0, 1 should get overriden.
    assert set(buf.sample(batch_size=100).test_field) == {2, 3, 4, 5}
