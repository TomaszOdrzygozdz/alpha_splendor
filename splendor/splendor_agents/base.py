from alpacka.tracing import State

from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace


class SplendorAgent:
    def __init__(self, deterministic: bool, name: str):
        self.deterministic = deterministic
        self.name = name

    def start(self, points_to_win):
        pass

    def act(self, clone_of_state: State, action_space: SplendorActionSpace):
        raise NotImplementedError

class DeterministicSplendorAgent(SplendorAgent):
    def __init__(self, name: str):
        super().__init__(True, name)

