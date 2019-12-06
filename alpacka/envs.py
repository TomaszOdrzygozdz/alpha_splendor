"""Environments."""

import gin
import gym
from gym.envs import classic_control
from gym_sokoban.envs import sokoban_env_fast
import numpy as np

from alpacka import data


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
        """Restores environment state, returns the observation."""
        raise NotImplementedError


@gin.configurable
class CartPole(classic_control.CartPoleEnv, ModelEnv):
    """CartPole with state clone/restore and returning a "solved" flag."""

    def __init__(self, solved_at=500, **kwargs):
        super().__init__(**kwargs)
        self._solved_at = solved_at
        self._step = None

    def reset(self):
        self._step = 0
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = self._step >= self._solved_at
        self._step += 1
        return (observation, reward, done, info)

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done, self._step)

    def restore_state(self, state):
        (state, self.steps_beyond_done, self._step) = state
        self.state = np.array(state)
        return self.state


@gin.configurable
class Sokoban(sokoban_env_fast.SokobanEnvFast, ModelEnv):
    """Sokoban with state clone/restore and returning a "solved" flag.

    Returns observations in one-hot encoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return observations as float32, so we don't have to cast them in the
        # network training pipeline.
        self.observation_space.dtype = np.float32

    def reset(self):
        return super().reset().astype(np.float32)

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        return (observation.astype(np.float32), reward, done, info)

    def clone_state(self):
        return self.clone_full_state()

    def restore_state(self, state):
        self.restore_full_state(state)
        return self.render(mode=self.mode)


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
