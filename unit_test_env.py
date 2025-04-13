from time import sleep
import gymnasium
from envs.racecar_gym_wrapper import TrackWrapper
import cv2

reward_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01,
    "steering_delta": 0.05,
    "steering_abs": 0.05
}

# set random seed from time
import time
import numpy as np
import envs.crafter as crafter

# env = crafter.Crafter("reward", (64,64))
# seed np to 0
np.random.seed(0)

# np.random.seed(int(time.time()))
# env = TrackWrapper(map_name='Columbia', render_mode='rgb_array_birds_eye', reward_config=rewards_config, include_state=False)
env = TrackWrapper(map_name='Austria', render_mode='human', reward_config=reward_config, include_state=False, logdir="train_logs/temp")
out = env.reset(options={"mode": "grid"})
total_rewards=0
STEER = 0.03
for i in range(2500):
    # action = env.action_space.sample()
    action = np.zeros(2)
    action[0]=0.01
    action[1]=1
    
    # if i>50 and i<100:
    #     action[1]=STEER
    # if i>100 and i<150:
    #     action[1]=-STEER
    # if i>200 and i<250:
    #     action[1]=STEER
    # if i>300 and i<350:
    #     action[1]=-STEER
    # if i>400 and i<450:
    #     action[1]=STEER
    # if i>500 and i<550:
    #     action[1]=-STEER
    
    # if i>560 and i<600:
    #     action[1]=0.1
    # if i>650 and i<800:
    #     action[1]=0.4
    # if i>1300 and i<1400:
    #     action[1]=0.1
    # if i>1500 and i<1600:
    #     action[1]=-0.1
    obs, rewards, done, info = env.step(action)
    # imshow obs["image"]
    cv2.imshow("Racecar", obs["image"])
    cv2.waitKey(1)
    total_rewards+=rewards
    render = env.render()
    # print(i, "Total rewards: ", total_rewards, "Reward: ", rewards)
    if done:
        break
    
env.close()