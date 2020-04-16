from splendor.envs.base import SplendorEnv

env = SplendorEnv()

is_done = False

initial_observation = env.reset()
while not is_done:
    try:
        action = env.action_space.sample()
    except:
        action = None
        print("Action is None")
    obs, reward, is_done, info = env.step(action)
    print(info)

