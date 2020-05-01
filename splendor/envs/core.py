import gin

from splendor.envs.base import SplendorEnv
from splendor.envs.mechanics.observation_generator import ObservationGenerator, PureObservationGenerator, \
    VectorizedObservationGenerator
from splendor.envs.mechanics.reward_evaluator import RewardEvaluator, OnlyVictory
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.splendor_agents.base import SplendorAgent


@gin.configurable
class OneSideSplendorEnv(SplendorEnv):
    def __init__(self,
                 splendor_agent: SplendorAgent,
                 points_to_win=15,
                 max_number_of_steps=120,
                 allow_reservations=True,
                 observation_space_generator = VectorizedObservationGenerator(),
                 reward_evaluator: RewardEvaluator = OnlyVictory(),
                 ):

        super().__init__(points_to_win, max_number_of_steps, allow_reservations, observation_space_generator, reward_evaluator)
        self.splendor_agent = splendor_agent
        self.splendor_agent.start(points_to_win)
        self.names = (('Real', self.splendor_agent.name))

    def clone_internal_state(self):
        return StateAsDict(self.internal_state).to_state()

    def step(self, action):
        self._step_env(action)
        if self.internal_state.is_done:
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info
        else:
            action = self.splendor_agent.act(self.clone_internal_state(), self.action_space)
            self._step_env(action)
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info
