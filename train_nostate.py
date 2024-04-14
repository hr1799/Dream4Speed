import json

EXPERIMENT_NAME = "map_Austria_reward_random_nostate"

# reward = if_collision * collision_reward + progress_reward * progress(%) + velocity_reward * sqrt(vx^2 + vy^2)
rewards_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01
}

def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    logdir = 'train_logs/dreamer/' + EXPERIMENT_NAME
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        # 'run.script': 'train_eval',
        'logdir': logdir,
        'run.train_ratio': 512,
        'run.log_every': 60,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': 'None',
        'decoder.mlp_keys': 'None',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'envs.length': 2000,
        'wrapper.length': 2000,
        # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(pattern=".*", logdir=logdir, config=config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    from dreamerv3.embodied.envs import from_gym
    from environments.racecar_gym_wrapper import TrackWrapper
    
    # env = crafter.Env()  # Replace this with your Gym env.
    env = TrackWrapper(map_name='Austria', render_mode='rgb_array_birds_eye', reward_config=rewards_config, include_state=False)
    
    env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    
    # Store the reward config as .yaml file in the logdir as rewards.yaml
    if not logdir.exists():
        logdir.mkdir(parents=True)
    rewards_file = logdir / 'rewards.json'
    with rewards_file.open('w') as f:
        json.dump(rewards_config, f)
    
    print("\n"*3)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
            config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
    embodied.run.eval_only(agent, env, logger, args)
    
    env.close()


if __name__ == '__main__':
    main()
