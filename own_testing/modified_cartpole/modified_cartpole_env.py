import math
import gym
from tensorflow import keras
import gin
from alpacka.agents import base
from alpacka.networks.keras import _make_inputs, _make_output_heads
from alpacka.utils.space import signature
from gym import spaces, logger
from gym.spaces import Tuple
from gym.utils import seeding
from gym.envs import classic_control
import numpy as np

@gin.configurable
def multi_input_network(network_signature, first_hidden_layer, second_hidden_layer, output_activation, output_zero_init, ):
    inputs = _make_inputs(network_signature.input)
    x = keras.layers.Concatenate()(list(inputs))
    x = keras.layers.Dense(first_hidden_layer, activation='relu')(x)
    x = keras.layers.Dense(first_hidden_layer, activation='relu')(x)
    x = keras.layers.Dense(second_hidden_layer, activation='relu')(x)
    outputs = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )

    return keras.Model(inputs=inputs, outputs=outputs)

class CartpoleActionSpace():
    def __init__(self):
        self.list_of_actions = ['left','right']

    def __len__(self):
        return len(self.list_of_actions)

    def __iter__(self):
        return self.list_of_actions.__iter__()

    def contains(self, action):
        return True if action in ['left', 'right'] else False

@gin.configurable
class MultiObservationCartpole(gym.Env):
    def __init__(self, solved_at=500, reward_scale=1., **kwargs):
        super().__init__(**kwargs)
        self.solved_at = solved_at
        self.reward_scale = reward_scale
        self._step = None
        self.local_cartpole = classic_control.CartPoleEnv()
        extra_input = np.array([10, 20, 30])
        self.action_space = CartpoleActionSpace()
        self.observation_space = spaces.Tuple( (self.local_cartpole.observation_space,
                                                spaces.Box(-extra_input, extra_input, dtype=np.float32)))

    def reset(self):
        self._step = 0
        original_observation = self.local_cartpole.reset()
        extra_part = np.zeros(3)
        return (original_observation, extra_part)

    def step(self, action):
        force_action = 0 if action == 'left' else 1
        (original_observation, reward, done, info) = self.local_cartpole.step(force_action)
        extra_part = np.zeros(3)
        observation = (original_observation, extra_part)
        info['solved'] = self._step >= self.solved_at
        self._step += 1
        return (observation, reward * self.reward_scale, done, info)

    def clone_state(self):
        return (tuple(self.local_cartpole.state), self._step)
        #return self.local_cartpole.state

    def restore_state(self, state):
        (state, self._step) = state
        self.local_cartpole.state = np.array(state)
        return self.local_cartpole.state