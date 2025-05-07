# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="run expert policy based on privileged information")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-reconstruction-v0', help="Name of the task.")
parser.add_argument("--num_traj", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--if_record", type=bool, default=True, help="if record data to lerobot")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import spinal_surgery
from isaaclab_tasks.utils import parse_env_cfg
import cProfile
import wandb
from spinal_surgery.lab.controllers.heuristic_reconstruction import HeuristicReconstruction
import tqdm
import numpy as np


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # init wandb
    wandb.init(project=args_cli.task, config=env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    square_size = 25

    heuristic_planner = HeuristicReconstruction(
        max_action=env.max_action, 
        action_scale=env.action_scale, 
        human_pos_2d_min=env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - square_size,
        human_pos_2d_max=env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + square_size,
        num_sections=2,
        total_steps=500,
        ratio=[0.05, 0.05, 0.05, 0.0], 
        device=env.sim.device)

    # reset environment
    obs, info = env.reset()
    for ep_index in tqdm.tqdm(range(args_cli.num_traj // args_cli.num_envs)):

        
        time_out = False
        step = 0
        while time_out is False:
            # run everything in inference mode
            actions = heuristic_planner.get_action(info, step)

            # apply actions
            obs, reward, term, time_out_tensor, info = env.step(actions)

            # change time_out to boolean
            step += 1
            time_out = torch.any(time_out_tensor).item()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    # close sim app
    simulation_app.close()
