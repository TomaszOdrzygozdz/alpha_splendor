from gym import Env
import gin

from splendor.envs.mechanics.observation_generator import ObservationGenerator, PureObservationGenerator, \
    VectorizedObservationGenerator
from splendor.envs.mechanics.reward_evaluator import RewardEvaluator, OnlyVictory
from splendor.envs.mechanics.splendor_action_space import SplendorActionSpace
from splendor.envs.mechanics.state import State
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.envs.utils.state_utils import statistics

@gin.configurable
class SplendorEnv(Env, ):
    def __init__(self,
                 points_to_win = 15,
                 max_number_of_steps = 120,
                 allow_reservations = True,
                 observation_space_generator: ObservationGenerator = None,
                 reward_evaluator: RewardEvaluator = OnlyVictory(),
                 observation_mode: str = 'vectorized'
                 ):

        if observation_space_generator is None:
            observation_generators = {'pure' : PureObservationGenerator(), 'vectorized' : VectorizedObservationGenerator()}
            assert observation_mode in observation_generators, 'Unknown observation mode'
            observation_space_generator = observation_generators[observation_mode]

        self.points_to_win = points_to_win
        self.max_number_of_steps = max_number_of_steps
        self.allow_reservations = allow_reservations
        self.observation_space_generator = observation_space_generator
        self.reward_evaluator = reward_evaluator

        self.action_space = SplendorActionSpace(self.allow_reservations)
        self.observation_space = self.observation_space_generator.return_observation_space()
        self.internal_state = State()
        self.names = None

    def clone_state(self):
        return StateAsDict(self.internal_state).to_state()

    def restore_state(self, state):
        copy_of_state = StateAsDict(state).to_state()
        self.internal_state = copy_of_state
        self.action_space.update(self.internal_state)

    def _check_if_done(self):
        if len(self.action_space) == 0:
            return True, 'Empty action space'
        for player in self.internal_state.list_of_players_hands:
            if player.number_of_my_points() >= self.points_to_win:
                return True, 'Win by points'
        return False, ''

    def _reset_env(self):
        """Resets the environment"""
        self.internal_state = State()
        self.action_space.update(self.internal_state)
        if self.names is not None:
            self.internal_state.set_names(self.names)

    def set_names(self, names):
        self.internal_state.set_names(names)

    def show_my_state(self):
        return (StateAsDict(self.internal_state))

    def game_statistics(self):
        game_statistics = {}
        for player in self.internal_state.list_of_players_hands:
            game_statistics[f'vp[{player.name}]'] = player.number_of_my_points()
            game_statistics[f'cards[{player.name}]'] = len(player.cards_possessed)
        return game_statistics

    def _step_env(self, action):
        """Performs the internal step on the environment"""
        assert not self.internal_state.is_done, 'Cannot take step on ended episode.'
        if action not in self.action_space.list_of_actions:
            print(f'Tried to take action {action}, while the action space consists of {self.action_space.list_of_actions} and \n'
                  f'while the state in MY_STATE = {StateAsDict(self.internal_state)}')
        assert action in self.action_space.list_of_actions or action is None, 'Wrong action'

        if self.internal_state.steps_taken_so_far > self.max_number_of_steps:
            self.internal_state.is_done = True

        else:
            if action is None:
                self.internal_state.is_done = True
                self.internal_state.winner = self.internal_state.other_players_hand().name
                self.internal_state.info['None_action'] = True
                self.internal_state.info['Done reason'] = 'Action None'
            else:
                self.internal_state.who_took_last_action = self.internal_state.active_players_hand().name
                action.execute(self.internal_state)
                self.action_space.update(self.internal_state)
                self.internal_state.is_done, reason = self._check_if_done()
                if self.internal_state.is_done:
                    self.internal_state.winner = self.internal_state.who_took_last_action
                    self.internal_state.info['None_action'] = False
                    self.internal_state.info['episode_length'] = self.internal_state.steps_taken_so_far+1
                    self.internal_state.info['Done reason'] = reason
                    #assert self.internal_state.list_of_players_hands[self.winner].number_of_my_points() >= self.points_to_win

            self.reward = self.reward_evaluator.evaluate(action, self.internal_state)

            if self.internal_state.is_done:
                self.internal_state.info['winner'] = self.internal_state.winner
                self.internal_state.info['who_took_last_action'] = self.internal_state.who_took_last_action
            self.internal_state.info['active'] = self.internal_state.active_player_id
            self.internal_state.info['step'] = self.internal_state.steps_taken_so_far
            self.internal_state.info['stats'] =  statistics(self.internal_state)
            self.internal_state.steps_taken_so_far += 1
        self.internal_state.info['additional_info'] = self.game_statistics()

    def _observation(self):
        return self.observation_space_generator.state_to_observation(self.internal_state)

    def step(self, action):
        self._step_env(action)
        return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info

    def reset(self):
        self._reset_env()
        return self._observation()