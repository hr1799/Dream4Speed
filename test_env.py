from robosuite.controllers import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from environments.env_make import make_env

env = make_env("Lift")

# print env.observation_space and env.action_space
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
# # simulate env for 10 steps
for i in range(1000):
    action = env.action_space.sample()  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on the screen

# close the env
env.close()
