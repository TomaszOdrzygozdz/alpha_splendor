import gin

from splendor.envs.base import SplendorEnv
from splendor.envs.mechanics.observation_generator import ObservationGenerator, PureObservationGenerator, \
    VectorizedObservationGenerator
from splendor.envs.mechanics.reward_evaluator import RewardEvaluator, OnlyVictory
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.splendor_agents.base import SplendorAgent


import numpy as np
import gym

from splendor.splendor_agents.greedy_heuristic import GreedyHeuristicAgent


@gin.configurable
class OneSideSplendorEnv(SplendorEnv):
    def __init__(self,
                 splendor_agent_class: SplendorAgent,
                 points_to_win=15,
                 max_number_of_steps=120,
                 allow_reservations=True,
                 observation_space_generator = VectorizedObservationGenerator(),
                 reward_evaluator: RewardEvaluator = OnlyVictory(),
                 real_player_name = 'Real'
                 ):

        super().__init__(points_to_win, max_number_of_steps, allow_reservations, observation_space_generator, reward_evaluator)
        self.splendor_agent = splendor_agent_class()
        self.splendor_agent.start(points_to_win)
        self.real_player_name = real_player_name
        self.names = ((self.real_player_name, self.splendor_agent.name))

    def clone_internal_state(self):
        return StateAsDict(self.internal_state).to_state()

    def step(self, action):
        self._step_env(action)
        if self.internal_state.is_done:
            if self.internal_state.info['winner'] == self.real_player_name:
                self.internal_state.info['solved'] = True
            if self.internal_state.info['winner'] != self.real_player_name:
                self.reward *= -1
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info
        else:
            action = self.splendor_agent.act(self.clone_internal_state(), self.action_space)
            self._step_env(action)
            if self.internal_state.info['winner'] == self.real_player_name:
                self.internal_state.info['solved'] = True
            if self.internal_state.info['winner'] != self.real_player_name:
                self.reward *= -1
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info

class DummyObservationGenerator:
    def state_to_observation(state):
        return np.array([1])

@gin.configurable
class DummyOneSideSplendor(OneSideSplendorEnv):
    def __init__(self):
        super().__init__(GreedyHeuristicAgent)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        self.observation_space_generator = DummyObservationGenerator()