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


RandomAgent = configure_agent(core.RandomAgent)  # pylint: disable=invalid-name
DeterministicMCTSAgent = configure_agent(  # pylint: disable=invalid-name
    deterministic_mcts.DeterministicMCTSAgent
)
ShootingAgent = configure_agent(shooting.ShootingAgent)  # pylint: disable=invalid-name
StochasticMCTSAgent = configure_agent(stochastic_mcts.StochasticMCTSAgent)  # pylint: disable=invalid-name

# Define target agents (Agent + Distribution).


@gin.configurable
class SoftmaxAgent(core.PolicyNetworkAgent):
    """Softmax agent, sampling actions from the categorical distribution."""

    def __init__(self, temperature=1.):
        """Initializes SoftmaxAgent.

        Args:
            temperature (float): Softmax temperature parameter.
        """
        super().__init__(
            pd=distributions.CategoricalPd(temperature=temperature))
