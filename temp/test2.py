from _ast import Tuple

import gym
import numpy as np
from alpacka.utils.space import signature
from gym.spaces import Box

env = gym.make('CartPole-v0')
env.reset()

signature(env.observation_space)
hh = np.array([-1,-2])
