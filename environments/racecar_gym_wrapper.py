import gymnasium
from gymnasium import Env
from racecar_gym.envs import gym_api
import numpy as np

class TrackWrapper():

    def __init__(
        self,
        map_name = 'Austria',
        render_mode = 'rgb_array_birds_eye',
        lidar_image_size = 128,
        lidar_image_resolution = 0.1, # meters per pixel
    ):
        if render_mode not in ['human', 'rgb_array_birds_eye', 'rgb_array_follow', 'rgb_array_lidar']:
            raise ValueError(f"Render mode {render_mode} not supported.")
        
        scenario = '/home/dreamerv3/environments/maps/' + map_name.lower() + '.yml'
        self.env = gymnasium.make('SingleAgentRaceEnv-v0',
                             scenario=scenario,
                             render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.reset()
        
    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        
        # obs_dict = self._flatten_obs(obs)
        obs_dict = obs
        
        return obs_dict, reward, done, info

    def render(self):
        return self.env.render()
    
    def _flatten_obs(self, obs):
        # pose (6,)
        # acceleration (6,)
        # velocity (6,)
        # lidar (1080,)
        # time ()
        
        # TODO hari
        
        obs_dict = {
            "state": state_vec,
            "image0": img0,
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
        # super().reset returns an OrderedDict
        ob_dict = self.env.reset(options=options)
        return self._flatten_obs(ob_dict)
    
    def close(self):
        self.env.close()