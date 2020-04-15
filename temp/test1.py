from gym import Env
from gym.spaces import Box, Tuple, Discrete
import numpy as np

class BabyEnv(Env):

    def __init__(self):
        self.observation_space = Tuple((Box(-1, 1, shape=(1,5)), Box(0, 1, shape=(1,7))))
        self.action_space = Discrete(2)
        self.state = [1,2]

    def sample(self):
        print(self.observation_space.sample())

    def step(self, action):
        del action
        obs = self.observation_space.sample()
        self.state = obs
        reward = np.random.uniform(-1,1)
        k = np.random.uniform(0, 1)
        is_done = True if k < 0.1 else False
        return obs, reward, is_done, {}

    def reset(self):
        return self.observation_space.sample()

