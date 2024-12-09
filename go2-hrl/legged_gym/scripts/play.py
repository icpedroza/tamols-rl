import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from legged_gym.utils import webviewer


def play(args):
    # web_viewer = webviewer.WebViewer()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # env.debug_viz = True

    # web_viewer.setup(env)


    # NOTE COMMENTED TODO UNCOMMENT
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # print("initialized policy")

    # num_feet = len(env.feet_indices)
    # target_positions = torch.zeros((env.num_envs, num_feet, 3), 
    #                              dtype=torch.float32, device=env.device)
    # # Set some example target positions
    # for i in range(env.num_envs):
    #     for j in range(num_feet):
    #         target_positions[i, j] = torch.tensor([0.3 * (j-1), 0.2 * (j-1), 0], 
    #                                             dtype=torch.float32, device=env.device)
    # env.update_target_positions(target_positions)
    
    # NOTE COMMENTED TODO UNCOMMENT
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # NOTE COMMENTED TODO UNCOMMENT
    # Comment out the loop that steps through the environment
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        print("stepping")
        obs, _, rews, dones, infos = env.step(actions.detach())
        
    # Draw planned footsteps
    planned_footsteps = [[0.3, 0.2, 0], [0.4, 0.3, 0], [0.5, 0.4, 0]]  # Example coordinates
    

    # Add spheres at the planned footsteps
    # planned_footsteps = target_positions.cpu().numpy()  # Convert to numpy for easier handling
    # for footstep in planned_footsteps[0]:  # Assuming we want to draw for the first environment
    #     # Assuming footstep is a list or array of [x, y, z] coordinates
    #     web_viewer.add_sphere(position=footstep, radius=0.05, color=[1, 0, 0])



    # web_viewer.render(fetch_results=True,
    #                   step_graphics=True,
    #                   render_all_camera_sensors=True,
    #                   wait_for_page_load=True)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
