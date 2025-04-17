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
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guided-surgery-v0', help="Name of the task.")
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
from isaaclab_tasks.utils import parse_env_cfg
import cProfile
import wandb
from spinal_surgery.lab.controllers.expert_guidance import ExpertGuidance
from spinal_surgery.lab.controllers.expert_surgery import ExpertSurgery


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

    # create expert
    if args_cli.task == "Isaac-robot-US-guidance-v0":
        expert = ExpertGuidance(env.max_action, env.action_scale, [0.02, 0.02, 0.3], env.sim.device)
    elif args_cli.task == "Isaac-robot-US-guided-surgery-v0":
        expert = ExpertSurgery(env.max_action, env.action_scale, [10.0, 10.0, 10.0, 0.1, 0.1, 0.1], env.sim.device)
    else:
        raise ValueError(f"Unsupported task: {args_cli.task}")

    # reset environment
    obs, info = env.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = expert.get_action(info)
            # apply actions
            obs, reward, term, time_out, info = env.step(actions)

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
