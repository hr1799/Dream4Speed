from time import sleep
import gymnasium
from environments.racecar_gym_wrapper import TrackWrapper
import cv2

env = TrackWrapper(map_name='Austria', render_mode='human')
out = env.reset()
print(out)
for i in range(10000):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    print(obs.keys())
    
env.close()
