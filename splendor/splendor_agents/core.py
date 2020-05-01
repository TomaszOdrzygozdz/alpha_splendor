from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state import State
from splendor.splendor_agents.base import DeterministicSplendorAgent


class DummyDeterministicAgent(DeterministicSplendorAgent):
    def __init__(self):
        super().__init__('Dummy deterministic agent')

    def act(self, state: State, action_space: SplendorActionSpace):
        for act in action_space:
            return act