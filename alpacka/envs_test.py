"""Tests for alpacka.envs."""

import pytest

from alpacka import envs


# TODO(koz4k): Test conformance with the ModelEnv API.


@pytest.mark.parametrize('env_class', [envs.CartPole, envs.Sokoban])
def test_shapes(env_class):
    env = env_class()
    obs = env.reset()
    assert obs.shape == env.observation_space.shape
    (obs, _, _, _) = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
