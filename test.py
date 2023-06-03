import gym
import env.environment as environment

env = environment.GolfEnv()
state = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    print('action: ', action)
    state, reward, done, info, _ = env.step(action)
    print('state: ', state)
    print('reward: ', reward)
    if done:
        print('done')
        break

env.close()