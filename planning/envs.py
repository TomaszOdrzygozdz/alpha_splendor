"""Environments."""

import gym
from gym.envs import classic_control

from planning import data

import numpy as np


class ModelEnv(gym.Env):
    """Environment interface used by model-based agents.

    This class defines an additional interface over gym.Env that is assumed by
    model-based agents. It's just for documentation purposes, doesn't have to be
    subclassed by envs used as models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores environment state."""
        raise NotImplementedError


class CartPole(classic_control.CartPoleEnv, ModelEnv):
    """CartPole with state clone/restore."""

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done)

    def restore_state(self, state):
        (state, self.steps_beyond_done) = state
        self.state = np.array(state)


class TransitionCollectorWrapper(gym.Wrapper):
    """Wrapper collecting transitions from the environment."""

    def __init__(self, env):
        super().__init__(env)
        self._transitions = []
        self._last_observation = None
        self.collect = True

    def reset(self):
        self._last_observation = super().reset()
        return self._last_observation

    def step(self, action):
        assert self._last_observation is not None, (
            'Environment must be reset before the first step().'
        )
        (next_observation, reward, done, info) = super().step(action)
        self.transitions.append(data.Transition(
            observation=self._last_observation,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation,
        ))
        self._last_observation = next_observation
        return (next_observation, reward, done, info)

    @property
    def transitions(self):
        return self._transitions
