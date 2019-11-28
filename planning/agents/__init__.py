"""Agents."""

import gin

from planning.agents import core
from planning.agents import mcts
from planning.agents.base import *


# Configure agents in this module to ensure they're accessible via the
# planning.agents.* namespace.
def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='planning.agents'
    )


RandomAgent = configure_agent(core.RandomAgent)  # pylint: disable=invalid-name
MCTSAgent = configure_agent(mcts.MCTSAgent)  # pylint: disable=invalid-name
