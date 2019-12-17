"""Environments."""

import gin
import gym
import numpy as np
from gym import wrappers
from gym.envs import classic_control
from gym_sokoban.envs import sokoban_env_fast


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


class ModelWrapper(gym.Wrapper):
    """Base class for wrappers intended for use with model-based environments.

    This class defines an additional interface over gym.Wrapper that is assumed
    by model-based agents. It's just for documentation purposes, doesn't have to
    be subclassed by wrappers used with models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        return self.env.clone_state()

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        return self.env.restore_state(state)


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


class TimeLimitWrapper(wrappers.TimeLimit, ModelWrapper):
    """Model-based TimeLimit gym.Env wrapper."""

    def clone_state(self):
        """Returns the current environment state."""
        assert self._elapsed_steps is not None, (
            'Environment must be reset before the first clone_state().'
        )

        return (super().clone_state(),
                ('TimeLimit._elapsed_step', self._elapsed_steps))

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        try:
            env_state, wrapper_state = state
            if wrapper_state[0] != 'TimeLimit._elapsed_step':
                raise ValueError()
            self._elapsed_steps = wrapper_state[1]
            state = env_state
        except (AttributeError, ValueError):
            self._elapsed_steps = 0

        return super().restore_state(state)
