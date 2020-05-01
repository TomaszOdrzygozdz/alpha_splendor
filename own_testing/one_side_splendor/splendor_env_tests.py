from splendor.envs.core import OneSideSplendorEnv
from splendor.envs.graphics.splendor_gui import SplendorGUI
from splendor.envs.mechanics.state_as_dict import StateAsDict
from splendor.splendor_agents.core import DummyDeterministicAgent
from splendor.splendor_agents.greedy_heuristic import GreedyHeuristicAgent

env = OneSideSplendorEnv(GreedyHeuristicAgent())
#env = OneSideSplendorEnv(DummyDeterministicAgent())
print(env.names)
env.reset()
print(env.names)

#gui = SplendorGUI()

def test():
    is_done = False
    for _ in range(3):
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