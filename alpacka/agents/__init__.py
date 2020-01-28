"""Agents."""

import gin

from alpacka.agents import core
from alpacka.agents import deterministic_mcts
from alpacka.agents import distributions
from alpacka.agents import shooting
from alpacka.agents import stochastic_mcts
from alpacka.agents import wrappers
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


class _DistributionAgent:
    """Base class for agents that use prob. dist. to sample action."""

    def __init__(self, distribution, with_critic, linear_annealing_kwargs):
        super().__init__()

        if with_critic:
            self._agent = ActorCriticAgent(distribution)
        else:
            self._agent = PolicyNetworkAgent(distribution)

        if linear_annealing_kwargs is not None:
            self._agent = wrappers.LinearAnnealingWrapper(
                self._agent,
                **linear_annealing_kwargs
            )

    def __getattr__(self, attr_name):
        return getattr(self._agent, attr_name)


@gin.configurable
class SoftmaxAgent(_DistributionAgent):
    """Softmax agent, sampling actions from the categorical distribution."""

    def __init__(self, temperature=1., with_critic=False,
                 linear_annealing_kwargs=None):
        """Initializes SoftmaxAgent.

        Args:
            temperature (float): Softmax temperature parameter.
            with_critic (bool): Run the Actor-Critic agent with a value network.
            linear_annealing_kwargs (dict): Temperature linear annealing
                schedule with keys: 'max_value', 'min_value', 'n_epochs',
                overrides temperature. See LinearAnnealingWrapper docstring.
                If None, then uses constant temperature value.
        """
        if linear_annealing_kwargs is not None:
            linear_annealing_kwargs['attr_name'] = 'distribution.temperature'

        super().__init__(
            distribution=distributions.CategoricalDistribution(
                temperature=temperature),
            with_critic=with_critic,
            linear_annealing_kwargs=linear_annealing_kwargs
        )


@gin.configurable
class EpsilonGreedyAgent(_DistributionAgent):
    """Softmax agent, sampling actions from the categorical distribution."""

    def __init__(self, epsilon=.05, with_critic=False,
                 linear_annealing_kwargs=None):
        """Initializes EpsilonGreedyAgent.

        Args:
            epsilon (float): Probability of taking random action.
            with_critic (bool): Run the Actor-Critic agent with a value network.
            linear_annealing_kwargs (dict): Epsilon linear annealing
                schedule with keys: 'max_value', 'min_value', 'n_epochs',
                overrides epsilon. See LinearAnnealingWrapper docstring.
                If None, then uses constant epsilon value.
        """
        if linear_annealing_kwargs is not None:
            linear_annealing_kwargs['attr_name'] = 'distribution.epsilon'

        super().__init__(
            distribution=distributions.EpsilonGreedyDistribution(
                epsilon=epsilon),
            with_critic=with_critic,
            linear_annealing_kwargs=linear_annealing_kwargs
        )
