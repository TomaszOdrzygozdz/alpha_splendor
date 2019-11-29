"""Environments."""

import gin
import gym
from gym.envs import classic_control
import numpy as np

from planning import data


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


@gin.configurable
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

    def reset(self, **kwargs):
        self._last_observation = super().reset(**kwargs)
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

        if done and 'solved' in info:
            # Some envs return a "solved" flag in the final step. We can use
            # it as a supervised target in value network training.
            # Rewrite the collected transitions, so we know they come from
            # a "solved" episode.
            self.transitions[:] = [
                transition._replace(solved=info['solved'])
                for transition in self.transitions
            ]

        return (next_observation, reward, done, info)

    @property
    def transitions(self):
        return self._transitions
