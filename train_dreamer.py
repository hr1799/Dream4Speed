import argparse
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import dreamer.tools as tools
from dreamer.dreamer import train_eval

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

# reward = if_collision * collision_reward + progress_reward * progress(%) + velocity_reward * sqrt(vx^2 + vy^2)
reward_config = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0.01,
    "steering_delta": 0.05,
    "steering_abs": 0.05
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, required=True)
    args, remaining = parser.parse_known_args()
    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "dreamer/configs.yaml").read_text()
    )

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)
    
    # update the above vars in final_config
    final_config.task = 'racecar_Austria'
    final_config.logdir = 'train_logs/' + args.expt_name
    final_config.run_train_ratio = 512
    final_config.batch_size = 16
    final_config.time_limit = 2000
    final_config.reward_config = reward_config
    final_config.envs = 1
    
    # save rewards_config to the logdir
    if not os.path.exists(final_config.logdir):
        os.makedirs(final_config.logdir)
    with open(final_config.logdir + '/reward_config.yaml', 'w') as f:
        yaml.dump(reward_config, f)
        
    train_eval(final_config)
