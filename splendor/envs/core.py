import gin

from splendor.envs.base import SplendorEnv
from splendor.envs.mechanics.observation_generator import ObservationGenerator, PureObservationGenerator, \
    VectorizedObservationGenerator
from splendor.envs.mechanics.reward_evaluator import RewardEvaluator, OnlyVictory

@gin.configurable
class OneSideSplendorEnv(SplendorEnv):
    def __init__(self,
                 points_to_win=15,
                 max_number_of_steps=120,
                 allow_reservations=True,
                 observation_space_generator = VectorizedObservationGenerator(),
                 observation_mode: str = 'vectorized',
                 reward_evaluator: RewardEvaluator = OnlyVictory()
                 ):

        super().__init__(points_to_win, max_number_of_steps, allow_reservations, observation_space_generator, reward_evaluator)
        self.internal_state.set_names(('Real', 'Internal'))

    def step(self, action):
        self._step_env(action)
        if self.internal_state.is_done:
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info
        else:
            action = self.action_space.list_of_actions[0] if len(self.action_space) > 0 else None
            self._step_env(action)
            return self._observation(), self.reward, self.internal_state.is_done, self.internal_state.info