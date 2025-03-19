
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run robot US guidance environment.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guidance-v0', help="Name of the task")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym

import spinal_surgery

def main():
    """Main function."""
    # create environment configuration
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            commands = torch.rand((args_cli.num_envs, 3)) * 2.0 - 1.0
            commands[:, 2] = (commands[:, 2] / 10)
            
            # step the environment
            obs, rew, terminated, truncated, info = env.step(commands)
            # print(rew)
            # print current orientation of pole
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()