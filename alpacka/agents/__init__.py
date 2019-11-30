"""Agents."""

import gin

from alpacka.agents import core
from alpacka.agents import mcts
from alpacka.agents.base import *


# Configure agents in this module to ensure they're accessible via the
# alpacka.agents.* namespace.
def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='alpacka.agents'
    )


RandomAgent = configure_agent(core.RandomAgent)  # pylint: disable=invalid-name
MCTSAgent = configure_agent(mcts.MCTSAgent)  # pylint: disable=invalid-name
