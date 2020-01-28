"""Test agent wrappers."""

from unittest import mock

import gym
import pytest

from alpacka import envs
from alpacka import testing
from alpacka.agents import base
from alpacka.agents import wrappers


@pytest.fixture
def agent():
    return mock.create_autospec(
        spec=base.Agent,
        instance=True
    )


@pytest.fixture
def env():
    return mock.create_autospec(
        spec=envs.CartPole,
        instance=True,
        action_space=mock.Mock(spec=gym.spaces.Discrete, n=2)
    )


def test_linear_annealing_wrapper(agent, env):
    # Set up
    attr_name = 'pied_piper'
    param_values = list(range(10, 0, -1))
    max_value = max(param_values)
    min_value = min(param_values)
    n_epochs = len(param_values)
    wrapped_agent = wrappers.LinearAnnealingWrapper(agent,
                                                    attr_name,
                                                    max_value,
                                                    min_value,
                                                    n_epochs)

    # Run & Test
    for epoch, x_value in enumerate(param_values):
        testing.run_without_suspensions(wrapped_agent.solve(env, epoch=epoch))
        assert getattr(agent, attr_name) == x_value
