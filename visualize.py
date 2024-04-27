import re
from collections import defaultdict
from functools import partial as bind
import dreamerv3

import embodied
import numpy as np

from environments.racecar_gym_wrapper import TrackWrapper

EXPERIMENT_NAME = "map_Austria_reward_random_nostate"

rewards_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01
}


def eval_only(make_agent, make_env, make_logger, args):
    assert args.from_checkpoint

    agent = make_agent()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    step = logger.step
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    policy_fps = embodied.FPS()

    @embodied.timer.section('log_step')
    def log_step(tran, worker):
        episode = episodes[worker]
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        if tran['is_first']:
            episode.reset()

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f'policy_{key}', tran[key], agg='stack')
        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg='sum')
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg='avg')
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg='max')

        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length') - 1,
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.num_envs)]
    driver = embodied.Driver(fns, args.driver_parallel)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(log_step)
    
    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.agent = agent
    checkpoint.load(logdir / 'checkpoint.ckpt', keys=['agent'])
    

    print('Start evaluation')
    policy = lambda *args: agent.policy(*args, mode='eval')
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)

def main():

    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update({
        **dreamerv3.Agent.configs['size12m'],
        'logdir': 'train_logs/dreamer/' + EXPERIMENT_NAME,
        'batch_size': 16,
        'enc.simple.minres': 8,
        'dec.simple.minres': 8,
        'run.num_envs': 8,
        'wrapper.length': 3000,
        
        'run.train_ratio': 512,
        'run.from_checkpoint': True,
        'run.driver_parallel': False,
        'run.num_envs': 1,
        'run.num_envs_eval': 1,
        
        'jax.debug': True,
        # 'jax.jit': False,
        'jax.prealloc': False, # If true, Preallocates 75% of the entire GPU memory
        'jax.platform': 'gpu',
        'jax.compute_dtype': 'bfloat16',
        'jax.profiler': False,
        'jax.checks': False,
    })
    
    config = embodied.Flags(config).parse()

    print('Logdir:', config.logdir)
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

    def make_agent(config):
        env = make_env(config, render_at_step=False)
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

    def make_env(config, env_id=0, render_at_step=True):
        from embodied.envs import from_gym
        # if render_at_step:
        #     render_mode = 'human'
        # else:
        #     render_mode = 'rgb_array_birds_eye'
        env = TrackWrapper(map_name='Austria', reward_config=rewards_config, include_state=False, render_at_step=render_at_step)
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
    
    eval_only(bind(make_agent, config), bind(make_env, config), bind(make_logger, config), args)


if __name__ == '__main__':
    main()
