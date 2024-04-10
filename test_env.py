from time import sleep
import gymnasium
from environments.racecar_gym_wrapper import TrackWrapper

env = TrackWrapper(map_name='Austria', render_mode='rgb_array_birds_eye')
for _ in range(5):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    
    # print keys in obs, rewards, done, info
    for k, v in obs.items():
        print(k, v.shape)
        
    # print(rewards)
    # print(done)
    # print(info)
    
    print("\n"*3)
    
env.close()
