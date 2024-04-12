from time import sleep
import gymnasium
from environments.racecar_gym_wrapper import TrackWrapper
import cv2

env = TrackWrapper(map_name='Austria', render_mode='human')
out = env.reset()
total_rewards=0
for i in range(10000):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    total_rewards+=rewards
    print(i, "Total rewards: ", total_rewards, "Done: ", done)
    if done:
        break
    
env.close()
