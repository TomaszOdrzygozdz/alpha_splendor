from alpacka.tracing import State

from splendor.envs.base import SplendorEnv
from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.splendor_agents.base import DeterministicSplendorAgent
from splendor.splendor_agents.utils.value_function import SplendorValueFunction


class GreedyHeuristicAgent(DeterministicSplendorAgent):
    def __init__(self):
        super().__init__('Greedy heuristic')
        self.env = SplendorEnv()

    def start(self, points_to_win):
        self.value_function = SplendorValueFunction(points_to_win)

    def act(self, state: State, action_space: SplendorActionSpace):
        def clone_state(state: State):
            return StateAsDict(state).to_state()


        my_player_name = state.active_players_hand().name

        def step_and_rewind(action):
            self.env.restore_state(clone_state(state))
            obs, rew, is_done, info = self.env.step(action)
            if is_done:
                if info['winner'] == my_player_name:
                    return float('inf')
            else:
                self.env.internal_state.change_active_player()
                return self.value_function.evaluate(self.env.internal_state)

        best_action = None
        best_eval = -float('inf')
        for action in action_space:
            eval = step_and_rewind(action)
            if eval > best_eval:
                best_eval = eval
                best_action = action

        return best_action
