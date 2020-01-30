"""Core agents."""

import asyncio

import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.utils import space as space_utils


class ActorCriticAgent(base.OnlineAgent):
    """Agent that uses value and policy networks to infer values and logits."""

    def __init__(self, distribution):
        """Initializes ActorCriticAgent.

        Args:
            distribution (ProbabilityDistribution): Probability distribution
                parameterized by the inferred logits to sample actions from and
                calculate statistics put into an agent info.
        """
        super().__init__()
        self._distribution = distribution

    def act(self, observation):
        batched_values, batched_logits = yield np.expand_dims(observation,
                                                              axis=0)
        values = np.squeeze(batched_values, axis=0)  # Removes batch dim.
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.

        action = self._distribution.sample(logits)
        agent_info = {'value': values}
        agent_info.update(self._distribution.compute_statistics(logits))

        return action, agent_info

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=(data.TensorSignature(shape=(1,)),
                    self._distribution.params_signature(action_space))
        )


class PolicyNetworkAgent(base.OnlineAgent):
    """Agent that uses a policy network to infer logits."""

    def __init__(self, distribution):
        """Initializes PolicyNetworkAgent.

        Args:
            distribution (ProbabilityDistribution): Probability distribution
                parameterized by the inferred logits to sample actions from and
                calculate statistics put into an agent info.
        """
        super().__init__()
        self._distribution = distribution

    def act(self, observation):
        batched_logits = yield np.expand_dims(observation, axis=0)
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.
        return (self._distribution.sample(logits),
                self._distribution.compute_statistics(logits))

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=self._distribution.params_signature(action_space),
        )


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return (self._action_space.sample(), {})
