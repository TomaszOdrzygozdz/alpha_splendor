"""CartPole env."""

from gym.envs import classic_control
import numpy as np

from alpacka.envs import base


class CartPole(classic_control.CartPoleEnv, base.ModelEnv):
    """CartPole with state clone/restore and returning a "solved" flag."""

    def __init__(self, solved_at=500, reward_scale=1., **kwargs):
        super().__init__(**kwargs)

        self.solved_at = solved_at
        self.reward_scale = reward_scale

        self._step = None

    def reset(self):
        self._step = 0
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = self._step >= self.solved_at
        self._step += 1
        return (observation, reward * self.reward_scale, done, info)

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done, self._step)

    def restore_state(self, state):
        (state, self.steps_beyond_done, self._step) = state
        self.state = np.array(state)
        return self.state
