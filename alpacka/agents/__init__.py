"""Agents."""

import gin

from alpacka.agents import core
from alpacka.agents import deterministic_mcts
from alpacka.agents import distributions
from alpacka.agents import shooting
from alpacka.agents import stochastic_mcts
from alpacka.agents.base import *


# Configure agents in this module to ensure they're accessible via the
# alpacka.agents.* namespace.
def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='alpacka.agents'
    )


ActorCriticAgent = configure_agent(core.ActorCriticAgent)  # pylint: disable=invalid-name
PolicyNetworkAgent = configure_agent(core.PolicyNetworkAgent)  # pylint: disable=invalid-name
RandomAgent = configure_agent(core.RandomAgent)  # pylint: disable=invalid-name
DeterministicMCTSAgent = configure_agent(  # pylint: disable=invalid-name
    deterministic_mcts.DeterministicMCTSAgent
)
ShootingAgent = configure_agent(shooting.ShootingAgent)  # pylint: disable=invalid-name
StochasticMCTSAgent = configure_agent(stochastic_mcts.StochasticMCTSAgent)  # pylint: disable=invalid-name

# Helper agents (Agent + Distribution).


class _PdAgent(OnlineAgent):
    """Base class for agents that use prob. dist. to sample action."""

    def __init__(self, pd, with_critic):
        super().__init__()

        if with_critic:
            self._agent = ActorCriticAgent(pd)
        else:
            self._agent = PolicyNetworkAgent(pd)

    def act(self, observation):
        return self._agent.act(observation)

    def network_signature(self, observation_space, action_space):
        return self._agent.network_signature(observation_space, action_space)


@gin.configurable
class SoftmaxAgent(_PdAgent):
    """Softmax agent, sampling actions from the categorical distribution."""

    def __init__(self, temperature=1., with_critic=False):
        """Initializes SoftmaxAgent.

        Args:
            temperature (float): Softmax temperature parameter.
            with_critic (bool): Run the Actor-Critic agent with a value network.
        """
        super().__init__(
            pd=distributions.CategoricalPd(temperature=temperature),
            with_critic=with_critic
        )


@gin.configurable
class EgreedyAgent(_PdAgent):
    """Softmax agent, sampling actions from the categorical distribution."""

    def __init__(self, epsilon=.05, with_critic=False):
        """Initializes EgreedyAgent.

        Args:
            epsilon (float): Probability of taking random action.
            with_critic (bool): Run the Actor-Critic agent with a value network.
        """
        super().__init__(
            pd=distributions.EgreedyPd(epsilon=epsilon),
            with_critic=with_critic
        )
