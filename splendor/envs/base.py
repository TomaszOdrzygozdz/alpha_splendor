from gym import Env
import gin

from splendor.envs.mechanics.observation_generator import ObservationGenerator, PureObservationGenerator
from splendor.envs.mechanics.reward_evaluator import RewardEvaluator, OnlyVictory
from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state import State
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.envs.utils.state_utils import statistics


class SplendorEnv(Env):
    def __init__(self,
            points_to_win = 15,
            max_number_of_steps = 120,
            allow_reservations = True,
            observation_space_generator: ObservationGenerator = PureObservationGenerator(),
            reward_evaluator: RewardEvaluator = OnlyVictory()):

        self.points_to_win = points_to_win
        self.max_number_of_steps = max_number_of_steps
        self.allow_reservations = allow_reservations
        self.observation_space_generator = observation_space_generator
        self.reward_evaluator = reward_evaluator

        self.steps_taken_so_far = 0
        self.action_space = SplendorActionSpace(self.allow_reservations)
        self.observation_space = self.observation_space_generator.return_observation_space()

        self.who_took_last_action = None #int describing id of the player or None
        self.winner = None #int describing id of the player or None
        self.is_done = False
        self.info = {}
        self.internal_state = State()

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

    def _reset_env(self):
        """Resets the environment"""
        self.is_done = False
        self.winner = False
        self.info = {}
        self.who_took_last_action = None
        self.steps_taken_so_far = 0
        self.action_space.update(self.internal_state)


    def _step_env(self, action):
        """Performs the internal step on the environment"""
        assert not self.is_done, 'Cannot take step on ended episode.'

        if self.steps_taken_so_far > self.max_number_of_steps:
            self.is_done = True

        else:
            if action is None:
                self.is_done = True
                self.winner = self.internal_state.other_players_hand().name
                self.info['None_action'] = True
            else:
                self.who_took_last_action = self.internal_state.active_players_hand().name
                action.execute(self.internal_state)
                self.action_space.update(self.internal_state)
                self.is_done = self._check_if_done()
                if self.is_done:
                    self.winner = self.who_took_last_action
                    self.info['None_action'] = False
                    self.info['episode_length'] = self.steps_taken_so_far+1
                    #assert self.internal_state.list_of_players_hands[self.winner].number_of_my_points() >= self.points_to_win

            self.reward = self.reward_evaluator.evaluate(action, self.internal_state, self.is_done,
                                                              self.who_took_last_action)

            if self.is_done:
                self.info['winner'] = self.winner
                self.info['who_took_last_action'] = self.who_took_last_action
            self.info['active'] = self.internal_state.active_player_id
            self.info['step'] = self.steps_taken_so_far
            self.info['stats'] =  statistics(self.internal_state)
            self.steps_taken_so_far += 1

    def _observation(self):
        return self.observation_space_generator.state_to_observation(self.internal_state)

    def step(self, action):
        self._step_env(action)
        return self._observation(), \
               self.reward, self.is_done, self.info

    def reset(self):
        self._reset_env()
        return self._observation()

class DualSplendorEnv(SplendorEnv):
    def __init__(self):
        super().__init__()

