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
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.
        return self._sample(logits), self._compute_statistics(logits)

    @staticmethod
    def network_signature(observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=data.TensorSignature(
                shape=(space_utils.max_size(action_space),)
            ),
        )

    def _compute_statistics(self, logits):
        """Computes softmax, log softmax and entropy with temperature."""
        w_logits = logits / self._temp
        c_logits = w_logits - np.max(w_logits, axis=-1, keepdims=True)
        e_logits = np.exp(c_logits)
        sum_e_logits = np.sum(e_logits, axis=-1, keepdims=True)

        prob = e_logits / sum_e_logits
        logp = c_logits - np.log(sum_e_logits)
        entropy = -np.sum(prob * logp, axis=-1)

        return {'prob': prob, 'logp': logp, 'entropy': entropy}

    def _sample(self, logits):
        """Sample from categorical distribution with temperature in log-space.

        See: https://stats.stackexchange.com/a/260248"""
        w_logits = logits / self._temp
        u = np.random.uniform(size=w_logits.shape)
        return np.argmax(w_logits - np.log(-np.log(u)), axis=-1)
