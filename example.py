import os
# set GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
EXPERIMENT_NAME = "map_Austria"

def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'logdir': 'train_logs/dreamer/' + EXPERIMENT_NAME,
        'run.train_ratio': 64,
        'run.log_every': 300,  # Seconds
        'batch_size': 4,
        'jax.prealloc': False,
        'encoder.mlp_keys': 'state',
        'decoder.mlp_keys': 'state',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
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

    import crafter
    from dreamerv3.embodied.envs import from_gym
    from environments.racecar_gym_wrapper import TrackWrapper
    
    # env = crafter.Env()  # Replace this with your Gym env.
    env = TrackWrapper(map_name='Austria', render_mode='rgb_array_birds_eye')
    
    env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    
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
