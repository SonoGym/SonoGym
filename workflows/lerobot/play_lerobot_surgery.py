"""Play multi-contact plan trajectories generated using LeRoBot.

This script demonstrates how to use the LeRoBot multi-contact planner and infer trajectories
from the current state of the simulation.

.. code-block:: bash

    python scripts/lerobot/play_lerobot.py ....

"""
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs import parser
from config import *




@parser.wrap()
def main(cfg: EvalPipelineConfig):
    ##
    # Launch Isaac Sim Simulator first.
    ##
    from isaaclab.app import AppLauncher

    # launch the simulator
    app_launcher = AppLauncher()
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch.nn.functional as F

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
    import wandb
    import numpy as np
    from spinal_surgery.tasks.robot_US_guided_surgery.robotic_US_guided_surgery import scene_cfg

    def get_surgery_policy_input(obs):
        image = obs['policy']['image']
        quat = obs['policy']['quat']
        pos = obs['policy']['pos']

        num_envs = image.shape[0]

        video = obs['policy']['image']

        video = torch.clamp(video, 0.0, 1.0)

        state = torch.cat((pos, quat), dim=-1)

        if scene_cfg['sim']['us'] == 'net':
            video = video
            obs = {
                "observation.images.slice_0": video[:, 0:1, :, :].repeat(1, 3, 1, 1),
                "observation.images.slice_1": video[:, 1:2, :, :].repeat(1, 3, 1, 1),
                "observation.images.slice_2": video[:, 2:3, :, :].repeat(1, 3, 1, 1),
                "observation.images.slice_3": video[:, 3:4, :, :].repeat(1, 3, 1, 1),
                "observation.images.slice_4": video[:, 4:5, :, :].repeat(1, 3, 1, 1),
                "observation.state": state,
            }
        else:
            # video = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False) # for diffusion
            video = video  # for diffusion
            obs = {
                "observation.images.slice_0": video[:, 0:5:2, :, :],
                "observation.images.slice_1": video[:, 5:10:2, :, :],
                "observation.images.slice_2": video[:, 10:15:2, :, :],
                "observation.images.slice_3": video[:, 15:20:2, :, :],
                "observation.images.slice_4": video[:, 20:25:2, :, :],
                "observation.state": state,
            }
        return obs

    env_cfg = parse_env_cfg(
        "Isaac-robot-US-guided-surgery-v0", 
        device="cuda", 
        num_envs=100, 
        use_fabric=True
    )

    env = gym.make("Isaac-robot-US-guided-surgery-v0", cfg=env_cfg, render_mode=None)

    # Load the dataset
    dataset_meta = LeRobotDatasetMetadata(repo_id="yunkao/SonoGym_lerobot_dataset",
                                          root="lerobot-dataset/Isaac-robot-US-guided-surgery-v0-single-new")
    # Create the policy
    print(cfg.policy)
    policy: PreTrainedPolicy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
    policy.eval()

    # Define simulation stepping
    sim_dt = env_cfg.sim.dt
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset all the environments
            obs, _ = env.reset()

            obs = get_surgery_policy_input(obs)
            # reset the policy
            policy.reset()

        # select action
        with torch.inference_mode():
            action = policy.select_action(obs)

        # print(action)

        # step the environment
        obs, _, _, _, _ = env.step(action)
        # rename keys of the dictionary to match lerobot
        obs = get_surgery_policy_input(obs)

        # update sim-time
        sim_time += sim_dt
        count += 1

    # close the environment
    env.close()
    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()
