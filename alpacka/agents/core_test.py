"""Tests for alpacka.agents.core."""

import collections

import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka import testing


def test_softmax_agent_network_signature():
    # Set up
    obs_space = gym.spaces.Box(low=0, high=255, shape=(7, 7), dtype=np.uint8)
    act_space = gym.spaces.Discrete(n=7)

    # Run
    signature = agents.SoftmaxAgent.network_signature(obs_space, act_space)

    # Test
    assert signature.input.shape == obs_space.shape
    assert signature.input.dtype == obs_space.dtype
    assert signature.output.shape == (act_space.n,)
    assert signature.output.dtype == np.float32


@pytest.mark.parametrize('logits',
                         [np.array([[3, 2, 1]]),
                          np.array([[1, 3, 2]]),
                          np.array([[2, 1, 3]])])
def test_softmax_agent_the_most_common_action_is_the_most_probable(logits):
    # Set up
    agent = agents.SoftmaxAgent()
    actions = []
    expected = np.argmax(logits)

    # Run
    for _ in range(100):
        action, _ = testing.run_with_constant_network_prediction(
            agent.act(np.zeros((7, 7))),
            logits
        )
        actions.append(action)

    (_, counts) = np.unique(actions, return_counts=True)
    most_common = np.argmax(counts)

    # Test
    assert most_common == expected


def test_softmax_agent_action_counts_for_different_temperature():
    # Set up
    low_temp_agent = agents.SoftmaxAgent(temperature=.5)
    high_temp_agent = agents.SoftmaxAgent(temperature=2.)
    low_temp_action_count = collections.defaultdict(int)
    high_temp_action_count = collections.defaultdict(int)
    logits = ((2, 1, 1, 1, 2), )  # Batch of size 1.

    # Run
    for agent, action_count in [
        (low_temp_agent, low_temp_action_count),
        (high_temp_agent, high_temp_action_count),
    ]:
        for _ in range(100):
            action, _ = testing.run_with_constant_network_prediction(
                agent.act(np.zeros((7, 7))),
                logits
            )
            action_count[action] += 1

    # Test
    assert low_temp_action_count[0] > high_temp_action_count[0]
    assert low_temp_action_count[1] < high_temp_action_count[1]
    assert low_temp_action_count[2] < high_temp_action_count[2]
    assert low_temp_action_count[3] < high_temp_action_count[3]
    assert low_temp_action_count[4] > high_temp_action_count[4]
