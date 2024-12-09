import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/Users/farazr/meam/anandir/go2-hrl/fetch/tamols')

torch.set_default_tensor_type(torch.FloatTensor)

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Create tensorboard writer with proper directory structure
    ## Writer code commented out as it requires rsl_rl changes I did not have repo access to

    # experiment_name = args.task  # Use task name as experiment name
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # log_dir = os.path.join('logs', experiment_name, current_time)
    # os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # writer = SummaryWriter(log_dir=log_dir)
    # print(f"Writing logs to {log_dir}")
    
    # try:
    #     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, writer=writer)
    #     ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # finally:
    #     writer.close()
    #     # Remove the env.close() call since it doesn't exist

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
