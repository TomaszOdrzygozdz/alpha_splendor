"""Core agents."""

import asyncio

import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.utils import space as space_utils


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return (self._action_space.sample(), {})


class SoftmaxAgent(base.OnlineAgent):
    """Softmax agent, sampling actions from the softmax dist.
    It evaluates a network to get logits."""

    def __init__(self, temperature=1.):
        """Initializes SoftmaxAgent.

        Args:
            temperature (float): Softmax temperature parameter.
        """
        super().__init__()
        self._temp = temperature

    def act(self, observation):
        batched_logits = yield np.expand_dims(observation, axis=0)
        logits = np.squeeze(batched_logits)  # Removes batch dim. of size 1
        pi = self._softmax(logits, self._temp)
        return (np.random.choice(pi.shape[0], p=pi), {})

    @staticmethod
    def network_signature(observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=data.TensorSignature(
                shape=(space_utils.max_size(action_space),)
            ),
        )

    @staticmethod
    def _softmax(x, temp=1.):
        """Computes a softmax function with temperature."""
        w_x = x / temp
        e_x = np.exp(w_x - np.max(w_x))
        return e_x / e_x.sum()
