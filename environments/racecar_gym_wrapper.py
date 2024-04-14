import gymnasium
from gymnasium import Env, spaces
from racecar_gym.envs import gym_api
import numpy as np

class TrackWrapper():

    def __init__(
        self,
        map_name = 'Austria',
        render_mode = 'rgb_array_birds_eye',
        lidar_image_size = 128,
        lidar_image_resolution = 0.1, # meters per pixel
        lidar_max_range = 10,
        lidar_angle_min_deg = -135,
        lidar_angle_increment_deg = 0.25,
        render_at_step=False,
        reward_config = None,
        include_state = True
    ):
        if render_mode not in ['human', 'rgb_array_birds_eye', 'rgb_array_follow', 'rgb_array_lidar']:
            raise ValueError(f"Render mode {render_mode} not supported.")

        self.reward_config = reward_config
        
        scenario = './environments/maps/' + map_name.lower() + '.yml'
        self.env = gymnasium.make('SingleAgentRaceEnv-v0',
                             scenario=scenario,
                             render_mode=render_mode)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict()
        if include_state:
            self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(13,))
        self.observation_space["image0"] = spaces.Box(low=0, high=1, shape=(lidar_image_size, lidar_image_size, 1))

        self.env.reset()

        self.lidar_image = None

        self.lidar_image_size = lidar_image_size
        self.lidar_image_resolution = lidar_image_resolution
        self.lidar_max_range = lidar_max_range
        self.lidar_angle_min_deg = lidar_angle_min_deg
        self.lidar_angle_increment_deg = lidar_angle_increment_deg
        self.render_at_step = render_at_step
        
    def step(self, action):
        action_to_env= {"motor": action[0], "steering": action[1]}
        obs, reward, done, _, privilaged_state = self.env.step(action_to_env)
        
        new_reward = reward
        if self.reward_config is not None:
            new_reward = reward*self.reward_config["progress_reward"]/100 + \
                            self.reward_config["velocity_reward"]*np.sqrt(privilaged_state["velocity"][0]**2 + privilaged_state["velocity"][1]**2)
                    
            if privilaged_state["wall_collision"] or np.any(privilaged_state["opponent_collisions"]):
                new_reward += self.reward_config["collision_reward"]
                done = True
        
        obs_dict = self._flatten_obs(obs)
        
        if self.render_at_step:
            self.env.render()
        self.step_count += 1
        
        return obs_dict, new_reward, done, {}

    def render(self):
        return self.env.render()
    
    def _flatten_obs(self, obs):
        # pose (6,)
        # acceleration (6,)
        # velocity (6,)
        # lidar (1080,)
        # time ()
        
        self.lidar_image = np.zeros((self.lidar_image_size, self.lidar_image_size, 1))
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
                self.lidar_image[y, x] = 1

        state_vec = np.concatenate([
                        [obs['time']],
                        obs['pose'],
                        obs['velocity']
                    ])
        
        obs_dict = {
            "state": state_vec,
            "image0": self.lidar_image,
        }
        
        return obs_dict
        
        
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
        ob_dict, _ = self.env.reset(options=options)
        self.step_count = 0
        return self._flatten_obs(ob_dict)
    
    def close(self):
        self.env.close()