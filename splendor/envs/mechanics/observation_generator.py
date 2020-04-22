from gym import Space

from splendor.envs.mechanics.abstract_observation import DeterministicObservation, StochasticObservation
from splendor.envs.mechanics.pure_observation_space import PureObservationSpace
from splendor.envs.mechanics.state import State


class ObservationGenerator:
    def state_to_observation(self, state : State):
        raise NotImplementedError

    def return_observation_space(self):
        try:
            return self.observation_space
        except:
            NotImplementedError('Observation space not specified')

class PureObservationGenerator(ObservationGenerator):
    def __init__(self, mode:str = 'deterministic'):
        super().__init__()
        self.observation_space = PureObservationSpace()
        self.mode = mode

    def state_to_observation(self, state: State):
        if self.mode == 'deterministic':
            return DeterministicObservation(state)
        if self.mode == 'stochastic':
            return StochasticObservation(state)
        else:
            raise ValueError('State to obsrvation mode not recognized.')

class TensorizedObservationGenerator(ObservationGenerator):
    pass


