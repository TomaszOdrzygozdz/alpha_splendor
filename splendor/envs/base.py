from abc import abstractmethod
from gym import Env
import gin

from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state import State
from splendor.envs.mechanics.state_as_dict import StateAsDict


@gin.configurable
class SplendorEnv(Env):
    def __init__(self,
            points_to_win = 15,
            max_number_of_steps = 120,
            allow_reservations = True):

        self.points_to_win = points_to_win
        self.max_number_of_steps = max_number_of_steps
        self.allow_reservations = allow_reservations
        self.action_space = SplendorActionSpace(self.allow_reservations)
        self.winner = None
        self.is_done = False
        self.internal_state = State()

    def reset(self):
        raise NotImplementedError

    def clone_state(self):
        return StateAsDict(self.internal_state).to_state()

    def restore_state(self, arg):
        if hasattr(arg, 'type'):
            if arg.type == 'dict':
                self.internal_state = arg.to_state()
            if arg.type == 'state':
                self.internal_state = arg
        else:
            raise ValueError(f'You must provide State or StateAsDict, received {type(arg)}')

    def _check_if_done(self):
        for player in self.internal_state.list_of_players_hands:
            if player.number_of_my_points() >= self.points_to_win:
                return True
        return False

    @abstractmethod
    def step(self, action):
        raise NotImplementedError
