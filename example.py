
# Set cuda visible devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("ATTENTION: Set cuda visible devices to ", os.environ["CUDA_VISIBLE_DEVICES"])

import warnings
from functools import partial as bind

import dreamerv3
import embodied
import json

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

from environments.racecar_gym_wrapper import TrackWrapper


# For Lift Robomimic
EXPERIMENT_NAME = "map_Austria_reward_random_nostate"
# reward = if_collision * collision_reward + progress_reward * progress(%) + velocity_reward * sqrt(vx^2 + vy^2)
rewards_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01
}

def main():

  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size25m'],
      'logdir': 'train_logs/dreamer/' + EXPERIMENT_NAME,
      'run.train_ratio': 512,
      'jax.prealloc': False, # If true, Preallocates 75% of the entire GPU memory
      'batch_size': 16,
      'enc.simple.minres': 8,
      'dec.simple.minres': 8,
      'run.num_envs': 8,
      'wrapper.length': 2000,
  })
  
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):
    from embodied.envs import from_gym
    env = TrackWrapper(map_name='Austria', render_mode='rgb_array_birds_eye', \
        reward_config=rewards_config, include_state=True)
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    return env

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )
  # Store the reward config as .yaml file in the logdir as rewards.yaml
  if not logdir.exists():
      logdir.mkdir(parents=True)
  rewards_file = logdir / 'rewards.json'
  with rewards_file.open('w') as f:
      json.dump(rewards_config, f)
    

  embodied.run.train(
      bind(make_agent, config),
      bind(make_replay, config),
      bind(make_env, config),
      bind(make_logger, config), args)


if __name__ == '__main__':
  main()
