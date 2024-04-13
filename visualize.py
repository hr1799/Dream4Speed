EXPERIMENT_NAME = "map_Austria"

import re

import dreamerv3.embodied as embodied
import numpy as np


def eval_only(agent, env, step, args):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    metrics = embodied.Metrics()
    print('Observation space:', env.obs_space)
    print('Action space:', env.act_space)

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy'])
    timer.wrap('env', env, ['step'])

    nonzeros = set()
    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix='stats')

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(logdir / 'checkpoint.ckpt', keys=['agent'])

    print('Start evaluation loop.')
    policy = lambda *args: agent.policy(*args, mode='eval')
    while step < args.steps:
        driver(policy, steps=100)


def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'run.script': 'train_eval',
        'logdir': 'train_logs/dreamer/' + EXPERIMENT_NAME,
        'run.train_ratio': 512,
        'run.log_every': 60,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': 'state',
        'decoder.mlp_keys': 'state',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'envs.length': 2000,
        'wrapper.length': 2000,
        # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    step = embodied.Counter()

    from dreamerv3.embodied.envs import from_gym
    from environments.racecar_gym_wrapper import TrackWrapper
    
    env = TrackWrapper(map_name='Austria', render_mode='human', render_at_step=True)
    
    env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    
    print("\n"*3)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length)
    
    eval_only(agent, env, step, args)
    
    env.close()


if __name__ == '__main__':
    main()
