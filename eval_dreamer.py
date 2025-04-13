import argparse
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import dreamer.tools as tools
from dreamer.dreamer import Dreamer
import torch
from envs.racecar_gym_wrapper import TrackWrapper
import numpy as np

REWARD_CONFIG_EVAL = {
    "collision_reward": -1,
    "progress_reward": 100,
    "velocity_reward": 0,
    "steering_delta": 0,
    "steering_abs": 0
}

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

class DreamerEval:
    def __init__(self, config, observation_space, action_space):
        tools.set_seed_everywhere(config.seed)
        tools.enable_deterministic_run()
        logdir = pathlib.Path(config.logdir).expanduser()
        config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        # Make Agent
        agent = Dreamer(
            observation_space,
            action_space,
            config,
            logger=None,
            dataset=None,
        ).to(config.device)
        
        agent.requires_grad_(requires_grad=False)
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

        self.state = None
        self.agent = agent
    
    def get_action(self, obs):
        # add batch dim to all keys in obs
        obs = {k: np.array([v]) for k, v in obs.items()}
        policy_output, self.state = self.agent(obs, reset=None, state=self.state, training=False)
        policy_output = {k: v[0] for k, v in policy_output.items()}
        return policy_output["action"].detach().cpu().numpy()

    def reset(self):
        self.state = None

if __name__ == "__main__":
    # Seed Torch and Numpy
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, required=True)
    parser.add_argument("--map_name", type=str, required=True)
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
    final_config.logdir = 'train_logs/' + args.expt_name
    
    map_name = args.map_name
    env = TrackWrapper(map_name=map_name, render_mode='human', # or 'rgb_array_birds_eye' or 'human'
        reward_config=REWARD_CONFIG_EVAL, include_state=False)
        
    agent_eval = DreamerEval(final_config, observation_space=env.observation_space, action_space=env.action_space)
    
    SAVE_DIR = "eval_logs/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    NUM_EVALS = 10
    env.reset()

    progresses = []
    times = []
    rewards = []

    eval_count = 0
    
    print()
    print("EVALUATING")
    print("Map:", map_name)
    print("Model:", args.expt_name)
    print("Num evals:", NUM_EVALS)
    print()
    
    for _ in range(NUM_EVALS):
        done = False
        obs = env.reset(options={"mode": "grid"})
        agent_eval.reset()
        total_reward_episode = 0
        itr = 0

        # Performance metrics
        progress = 0
        time_taken = 0

        print("\nStarting eval", eval_count)
        while not done:
            itr += 1
            if itr % 500 == 0:
                print("Step:", itr)
            action = agent_eval.get_action(obs)
            obs, reward, done, privilaged_state = env.step(action)
            
            progress = max(privilaged_state["progress"], progress)
            time_taken = privilaged_state["time"]

            total_reward_episode += reward
        
        print(f"Completed eval {eval_count} with reward:", total_reward_episode, "Progress:", progress, "Time taken:", time_taken)
        progresses.append(progress)
        times.append(time_taken)
        rewards.append(total_reward_episode)
        
        eval_count += 1
    
    # Save the performance metrics in CSV
    import csv
    with open(f"eval_logs/{args.expt_name}_{map_name}.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["#", "Progress", "Time", "Reward"])
        for i in range(NUM_EVALS):
            writer.writerow([i, progresses[i], times[i], rewards[i]])
