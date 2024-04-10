from time import sleep
import gymnasium
from environments.racecar_gym_wrapper import TrackWrapper
import cv2

env = TrackWrapper(map_name='Austria', render_mode='human')
for _ in range(10000):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)

    lidar_img = obs['image0']
    cv2.imshow('lidar', lidar_img)
    cv2.waitKey(0)
    
    # print keys in obs, rewards, done, info
    for k, v in obs.items():
        print(k, v.shape)
        
    images = env.render()
    # print(rewards)
    # print(done)
    # print(info)
    
    print("\n"*3)
    
env.close()
