import gin

from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state import State
from splendor.splendor_agents.base import DeterministicSplendorAgent

@gin.configurable
class DummyDeterministicAgent(DeterministicSplendorAgent):
    def __init__(self):
        super().__init__('Dummy deterministic agent')

    def act(self, state: State, action_space: SplendorActionSpace):
        for act in action_space:
            return act

@gin.configurable
class RandomStochasticAgent(DeterministicSplendorAgent):
    def __init__(self):
        super().__init__('Dummy deterministic agent')

    def act(self, state: State, action_space: SplendorActionSpace):
        return action_space.sample()