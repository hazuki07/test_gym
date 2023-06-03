from gym.envs.registration import register

register(
    id='GolfEnv-v0',
    entry_point='env.environment:GolfEnv'
)