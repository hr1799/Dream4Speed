import gymnasium
from gymnasium import Env, spaces
from racecar_gym.envs import gym_api
import numpy as np
import cv2
import time

class TrackWrapper():
    metadata = {}
    def __init__(
        self,
        map_name = 'Austria',
        render_mode = 'rgb_array_birds_eye',
        lidar_image_size = 128,
        lidar_image_resolution = 0.1, # meters per pixel
        lidar_max_range = 10,
        lidar_angle_min_deg = -135,
        lidar_angle_increment_deg = 0.25,
        reward_config = None,
        include_state = True,
        should_save_video = False,
        logdir = None # to save action csvs
    ):
        if render_mode not in ['human', 'rgb_array_birds_eye', 'rgb_array_follow']:
            raise ValueError(f"Render mode {render_mode} not supported.")
        
        self.map_name = map_name
        self.should_save_video = should_save_video
        self.rendered_imgs = []
        self.video_count = 0

        self.reward_config = reward_config

        scenario = '/home/racecar_gym/scenarios/' + map_name.lower() + '.yml'
        self.env = gymnasium.make('SingleAgentRaceEnv-v0',
                             scenario=scenario,
                             render_mode=render_mode)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict()
        self.include_state = include_state
        if include_state:
            self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        else:
            self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space["image"] = spaces.Box(low=0, high=255, shape=(lidar_image_size, lidar_image_size, 3), dtype=np.uint8)

        self.reward_range = [-np.inf, np.inf]
        self.env.reset()

        self.lidar_image_size = lidar_image_size
        self.lidar_image_resolution = lidar_image_resolution
        self.lidar_max_range = lidar_max_range
        self.lidar_angle_min_deg = lidar_angle_min_deg
        self.lidar_angle_increment_deg = lidar_angle_increment_deg
        self.logdir = logdir
        
    def step(self, action):
        steering_rate = action[1] * np.pi
        dt = 0.01
        steering = self.prev_steering + steering_rate * dt
        # clip steering to -0.42 to 0.42
        steering = np.clip(steering, -0.42, 0.42)
        self.prev_steering = steering

        steering /= 0.42

        print(steering)
        # Append actions to list for logging if logdir is provided
        if self.logdir is not None:
            self.actions.append([action[0], action[1], steering]) # speed, steering rate, steering angle (unnomalized)
            
        action_to_env= {"speed": action[0], "steering": steering}
        obs, reward, done, _, privilaged_state = self.env.step(action_to_env)
        
        new_reward = reward
        if self.reward_config is not None:
            new_reward = reward*self.reward_config["progress_reward"]/100 + \
                            self.reward_config["velocity_reward"]*np.sqrt(privilaged_state["velocity"][0]**2 + privilaged_state["velocity"][1]**2)
                    
            if privilaged_state["wall_collision"] or np.any(privilaged_state["opponent_collisions"]):
                new_reward += self.reward_config["collision_reward"]
                done = True
            
            # negative reward for steering
            steer_reward = self.reward_config["steering_abs"]*(action[1]**2)
            new_reward -= steer_reward
        
        obs_dict = self._flatten_obs(obs)
        
        if self.should_save_video:
            img = self.render()
            if img is not None:
                self.rendered_imgs.append(img)
            
        self.step_count += 1
        
        obs_dict["is_first"] = False
        obs_dict["is_last"] = done
        obs_dict["is_terminal"] = done
        
        if done and self.logdir is not None:
            # dump actions as csv with timestamp
            all_actions = np.array(self.actions)
            filename = "actions_" + str(time.time()) + ".csv"
            np.savetxt(self.logdir + "/" + filename, all_actions, delimiter=",")
            
            # reset actions list
            self.actions = []
        
        return obs_dict, new_reward, done, privilaged_state

    def render(self):
        return self.env.render()
    
    def _flatten_obs(self, obs):
        # pose (6,)
        # acceleration (6,)
        # velocity (6,)
        # lidar (1080,)
        # time ()
        
        self.lidar_image = np.zeros((self.lidar_image_size, self.lidar_image_size, 3), dtype=np.uint8)
        lidar_obs = obs['lidar']

        for i in range(len(lidar_obs)):
            if lidar_obs[i] > self.lidar_max_range * 0.95:
                continue
            if lidar_obs[i] < 0:
                continue
            
            angle = self.lidar_angle_min_deg + i * self.lidar_angle_increment_deg
            angle_rad = np.deg2rad(angle)
            x = int(lidar_obs[i] * np.cos(angle_rad) / self.lidar_image_resolution)
            y = int(lidar_obs[i] * np.sin(angle_rad) / self.lidar_image_resolution + self.lidar_image_size / 2)
            if x >= 0 and x < self.lidar_image_size and y >= 0 and y < self.lidar_image_size:
                self.lidar_image[y, x, :] = 255

        if self.include_state:
            state_vec = np.concatenate([
                            [obs['time']],
                            obs['pose'],
                            obs['velocity']
                        ])
        else:
            state_vec = np.zeros(1, dtype=np.float64)
        
        obs_dict = {
            "state": state_vec,
            "image": self.lidar_image,
        }
        
        return obs_dict
    
    def save_video(self, video_name):
        if not self.should_save_video:
            print("Cannot save video as should_save_video is False")
            return
        print(f"Saving video as {video_name}")
        frame_height, frame_width, _ = self.rendered_imgs[0].shape
        # save rendered images as video mp4
        out = cv2.VideoWriter(video_name, \
                                cv2.VideoWriter_fourcc(*'mp4v'), 100, (frame_width, frame_height))
        for img in self.rendered_imgs:
            out.write(img)
        out.release()
        print(f"Video saved as video_{self.video_count}.mp4")
        
    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        if options is None:
            options = {"mode": "random"}
        
        # Reset variables
        self.step_count = 0

        if self.should_save_video and len(self.rendered_imgs) > 0:
            self.rendered_imgs = []
            self.video_count += 1
        
        self.prev_steering = 0
        self.actions = []
        
        ob_dict, _ = self.env.reset(options=options)
        ob_dict = self._flatten_obs(ob_dict)
        ob_dict["is_first"] = True
        ob_dict["is_last"] = False
        ob_dict["is_terminal"] = False
        
        return ob_dict
    
    def close(self):
        self.env.close()