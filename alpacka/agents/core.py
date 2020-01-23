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


class PolicyNetworkAgent(base.OnlineAgent):
    """Agent that uses a policy network to infer logits."""

    def __init__(self, pd):
        """Initializes PolicyNetworkAgent.

        Args:
            pd (ProbabilityDistribution): Probability distribution parameterized
                by the inferred logits to sample actions from and calculate
                statistics put into an agent info.
        """
        super().__init__()
        self._pd = pd

    def act(self, observation):
        batched_logits = yield np.expand_dims(observation, axis=0)
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.
        return self._pd.sample(logits), self._pd.compute_statistics(logits)

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=self._pd.params_signature(action_space),
        )
