from splendor.envs.core import OneSideSplendorEnv
from splendor.envs.graphics.splendor_gui import SplendorGUI
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.splendor_agents.core import DummyDeterministicAgent, RandomStochasticAgent
from splendor.splendor_agents.greedy_heuristic import GreedyHeuristicAgent

#env = OneSideSplendorEnv(RandomStochasticAgent())
#env = OneSideSplendorEnv(DummyDeterministicAgent())
env = OneSideSplendorEnv(GreedyHeuristicAgent())
env.reset()

#gui = SplendorGUI()

def test():
    is_done = False
    for _ in range(10):
        while not is_done:
            print('.', end='')
            action=env.action_space.sample()
            obs, rew, is_done, info = env.step(action)
        print(env.internal_state.winner)
        env.reset()
        is_done = False

test()
# import cProfile
# cProfile.run('test()', filename='tutek.prof')