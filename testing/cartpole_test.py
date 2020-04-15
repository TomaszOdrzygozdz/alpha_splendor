import gym

env = gym.make('CartPole-v0')
env.reset()

observation, reward, done, info = env.step(1)

print(observation)

env.close()