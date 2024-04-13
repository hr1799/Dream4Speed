from time import sleep
import gymnasium
from environments.racecar_gym_wrapper import TrackWrapper
import cv2

rewards_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01
}

# set random seed from time
import time
import numpy as np
np.random.seed(int(time.time()))
env = TrackWrapper(map_name='Austria', render_mode='human', reward_config=rewards_config)
out = env.reset()
total_rewards=0
for i in range(10000):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    total_rewards+=rewards
    print(i, "Total rewards: ", total_rewards, "Reward: ", rewards)
    if done:
        break
    sleep(0.1)
    
env.close()
