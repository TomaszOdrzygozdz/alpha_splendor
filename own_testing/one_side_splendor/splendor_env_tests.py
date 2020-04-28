from splendor.envs.core import OneSideSplendorEnv
from splendor.envs.graphics.splendor_gui import SplendorGUI
from splendor.envs.mechanics.state_as_dict import StateAsDict

env = OneSideSplendorEnv()
env.reset()

gui = SplendorGUI()

is_done = False
for _ in range(10):
    while not is_done:
        print('.', end='')
        gui.draw_state(env.internal_state)
        action = gui.read_action()
        obs, rew, is_done, info = env.step(action)
    print(env.internal_state.winner)
    env.reset()
    is_done = False
