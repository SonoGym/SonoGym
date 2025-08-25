# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="run expert policy based on privileged information"
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-robot-US-guided-surgery-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_traj", type=int, default=2048, help="Number of environments to simulate."
)
parser.add_argument(
    "--if_record", type=bool, default=True, help="if record data to lerobot"
)

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
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from spinal_surgery.tasks.robot_US_guidance.cfgs.lerobot_cfg import *
from spinal_surgery.tasks.robot_US_guided_surgery.cfgs.lerobot_cfg import *
import tqdm
import numpy as np
import cv2
from spinal_surgery.tasks.robot_US_guidance.robotic_US_guidance import scene_cfg
from spinal_surgery.tasks.robot_US_guided_surgery.robotic_US_guided_surgery import (
    scene_cfg as scene_cfg_surgery,
)

guidance_us_mode = scene_cfg["sim"]["us"]
surgery_us_mode = scene_cfg_surgery["sim"]["us"]


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # init wandb
    wandb.init(project=args_cli.task, config=env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create expert
    if args_cli.task == "Isaac-robot-US-guidance-v0":
        expert = ExpertGuidance(
            env.max_action, env.action_scale, [0.02, 0.02, 0.6], env.sim.device
        )
        features = GUIDANCE_FEATURES
    elif args_cli.task == "Isaac-robot-US-guided-surgery-v0":
        expert = ExpertSurgery(
            env.max_action,
            env.action_scale,
            [20.0, 20.0, 20.0, 0.1, 0.1, 0.1],
            env.sim.device,
        )
        features = SURGERY_FEATURES
    else:
        raise ValueError(f"Unsupported task: {args_cli.task}")

    if args_cli.if_record:
        # create lerobot dataset
        # lerobot_dataset = LeRobotDataset.create(
        #     repo_id="yunkao/uspine",
        #     fps=int(1 / env_cfg.sim.dt),
        #     root="/home/yunkao/git/IsaacLabExtensionTemplate/lerobot-dataset/"
        #     + args_cli.task,
        #     robot_type="kuka-med",
        #     features=features,
        # )
        lerobot_dataset = LeRobotDataset(
            repo_id="yunkao/uspine",
            root="/home/yunkao/git/IsaacLabExtensionTemplate/lerobot-dataset/Isaac-robot-US-guided-surgery-v0",
        )

    for ep_index in tqdm.tqdm(range(args_cli.num_traj // args_cli.num_envs)):

        # reset environment
        obs, info = env.reset()
        time_out = False

        stacked_frame_list = []
        while time_out is False:
            # run everything in inference mode
            actions = expert.get_action(info)
            # apply actions
            obs, reward, term, time_out, info = env.step(actions)

            # record data
            if args_cli.task == "Isaac-robot-US-guidance-v0":
                observation = obs["policy"].cpu().numpy()
                action = actions.cpu().numpy()
                if guidance_us_mode == "net":
                    observation *= 6

                stacked_frame = {
                    "observation": np.clip(
                        observation, 0.0, 1.0
                    ),  # *30 for 0.02 scale net guidance,
                    "action": action,
                    "task": GUIDANCE_TASK,
                }
                stacked_frame_list.append(stacked_frame)
            else:
                obs_img = obs["policy"]["image"].squeeze(0).cpu().numpy()
                obs_pos = obs["policy"]["pos"].squeeze(0).cpu().numpy()
                obs_quat = obs["policy"]["quat"].squeeze(0).cpu().numpy()
                action = actions.squeeze(0).cpu().numpy()
                 
                stacked_frame = {
                    "observation.ultrasound": np.clip(
                        obs_img, 0.0, 1.0
                    ),  # 5 for 0.02 scale net surgery, *1 for 0.02 scale model based
                    "observation.pos": obs_pos,
                    "observation.quat": obs_quat,
                    "action": action,
                    "task": SURGERY_TASK,
                }
                stacked_frame_list.append(stacked_frame)

            # change time_out to boolean
            time_out = torch.any(time_out).item()

        if args_cli.if_record:
            for ep in range(args_cli.num_envs):
                for i in range(len(stacked_frame_list)):
                    if args_cli.task == "Isaac-robot-US-guidance-v0":
                        frame = {
                            "observation.images": np.repeat(
                                stacked_frame_list[i]["observation"][ep, ...], 3, axis=0
                            ).transpose(1, 2, 0),
                            "observation.state": np.zeros((2,), dtype=np.float32),
                            "action": stacked_frame_list[i]["action"][ep, :],
                            "task": GUIDANCE_TASK,
                        }
                    else:
                        state = np.concatenate(
                            (
                                stacked_frame_list[i]["observation.pos"][ep, :],
                                stacked_frame_list[i]["observation.quat"][ep, :],
                            ),
                            axis=0,
                        )
                        video = stacked_frame_list[i]["observation.ultrasound"][
                            ep, ...
                        ]  # (25, 50, 37)
                        if video.shape[0] == 25:
                            # video = (video * 768).astype(np.uint8)  # only for diffusion
                            # print(np.max(video), np.min(video))
                            # frame = {
                            #         "observation.images.slice_0": cv2.resize(video[0:5:2, :, :].transpose(1, 2, 0), (256, 256)),
                            #         "observation.images.slice_1": cv2.resize(video[5:10:2, :, :].transpose(1, 2, 0), (256, 256)),
                            #         "observation.images.slice_2": cv2.resize(video[10:15:2, :, :].transpose(1, 2, 0), (256, 256)),
                            #         "observation.images.slice_3": cv2.resize(video[15:20:2, :, :].transpose(1, 2, 0), (256, 256)),
                            #         "observation.images.slice_4": cv2.resize(video[20:25:2, :, :].transpose(1, 2, 0), (256, 256)),
                            #         "observation.state": state,
                            #         "action": stacked_frame_list[i]['action'][ep, :],
                            #         'task': SURGERY_TASK,
                            # }
                            frame = {
                                "observation.images.slice_0": video[
                                    0:5:2, :, :
                                ].transpose(1, 2, 0),
                                "observation.images.slice_1": video[
                                    5:10:2, :, :
                                ].transpose(1, 2, 0),
                                "observation.images.slice_2": video[
                                    10:15:2, :, :
                                ].transpose(1, 2, 0),
                                "observation.images.slice_3": video[
                                    15:20:2, :, :
                                ].transpose(1, 2, 0),
                                "observation.images.slice_4": video[
                                    20:25:2, :, :
                                ].transpose(1, 2, 0),
                                "observation.state": state,
                                "action": stacked_frame_list[i]["action"][ep, :],
                                "task": SURGERY_TASK,
                            }
                        else:
                            frame = {
                                "observation.images.slice_0": np.repeat(
                                    video[0:1, :, :].transpose(1, 2, 0), 3, axis=-1
                                ),
                                "observation.images.slice_1": np.repeat(
                                    video[1:2, :, :].transpose(1, 2, 0), 3, axis=-1
                                ),
                                "observation.images.slice_2": np.repeat(
                                    video[2:3, :, :].transpose(1, 2, 0), 3, axis=-1
                                ),
                                "observation.images.slice_3": np.repeat(
                                    video[3:4, :, :].transpose(1, 2, 0), 3, axis=-1
                                ),
                                "observation.images.slice_4": np.repeat(
                                    video[4:5, :, :].transpose(1, 2, 0), 3, axis=-1
                                ),
                                "observation.state": state,
                                "action": stacked_frame_list[i]["action"][ep, :],
                                "task": SURGERY_TASK,
                            }
                    # add to frame
                    lerobot_dataset.add_frame(frame)
                # save episode
                lerobot_dataset.save_episode()
                lerobot_dataset.clear_episode_buffer()

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
