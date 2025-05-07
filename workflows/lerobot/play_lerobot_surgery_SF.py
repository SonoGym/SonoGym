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

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
    import wandb
    import numpy as np
    from spinal_surgery.tasks.robot_US_guided_surgery.robotic_US_guided_surgery import scene_cfg

    from spinal_surgery.lab.agents.skrl_safety_filter_agent import SafetyFilterPPO, SPPO_DEFAULT_CONFIG
    from spinal_surgery.lab.agents.skrl_actor_critic import SharedModel, QNet
    from skrl.memories.torch import RandomMemory
    from skrl.resources.schedulers.torch.kl_adaptive import KLAdaptiveLR
    from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

    def get_surgery_policy_input(obs):
        # print(obs)
        image = obs['image']
        quat = obs['quat']
        pos = obs['pos']

        num_envs = image.shape[0]

        video = obs['image']

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
        num_envs=32, 
        use_fabric=True
    )

    env = gym.make("Isaac-robot-US-guided-surgery-v0", cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework='torch')

    # Load the dataset
    dataset_meta = LeRobotDatasetMetadata(repo_id="yunkao/us-guidance",
                                          root="/home/yunkao/git/IsaacLabExtensionTemplate/lerobot-dataset/Isaac-robot-US-guided-surgery-v0-single")
    # Create the policy
    print(cfg.policy)
    policy: PreTrainedPolicy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
    policy.eval()

    # Define simulation stepping
    sim_dt = env_cfg.sim.dt
    sim_time = 0.0
    count = 0

    # resume sppo
    resume_path = "/home/yunkao/git/IsaacLabExtensionTemplate/logs/experiments/us-guided-surgery/human_5/model-based-sim/SaferPlan/2025-05-01_17-48-43_ppol_torch_SPPO_default_US_net/checkpoints/agent_200000.pt"
    agent_cfg = load_cfg_from_registry("Isaac-robot-US-guided-surgery-v0", "skrl_sppo_cfg_entry_point")
    models = {}
    device = env_cfg.sim.device
    models["policy"] = SharedModel(env.observation_space, env.action_space, device, clip_actions=False)
    models["value"] = models["policy"]
    models["cost_critic"] = QNet(env.observation_space, env.action_space, device)
    if agent_cfg['learning_rate_scheduler'] == "KLAdaptiveLR":
        agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    if agent_cfg['value_preprocessor'] == "RunningStandardScaler":
        agent_cfg["value_preprocessor"] = RunningStandardScaler
        agent_cfg["value_preprocessor_kwargs"] = {'size': 1}

    memory = RandomMemory(memory_size=agent_cfg['rollouts'], num_envs=env.num_envs, device=device, replacement=False)

    sppo_agent = SafetyFilterPPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    sppo_agent.load(resume_path)

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset all the environments
            obs, _ = env.reset()

            lerobot_obs = get_surgery_policy_input(
                sppo_agent.models['policy'].tensor_to_space(obs, sppo_agent.models['policy'].observation_space)
            )
            # reset the policy
            policy.reset()

        # select action
        with torch.inference_mode():
            action = policy.select_action(lerobot_obs)

            # apply safety filter
            inputs = {'states': obs, 'taken_actions': action}
            pred_cost = sppo_agent.cost_critic.act(inputs, role="cost_critic")[0].reshape((-1,))
            action[pred_cost > 0.01, :] = 0.0

        # step the environment
        obs, _, _, _, _ = env.step(action)
        # rename keys of the dictionary to match lerobot
        lerobot_obs = get_surgery_policy_input(
            sppo_agent.models['policy'].tensor_to_space(obs, sppo_agent.models['policy'].observation_space)
        )

        # update sim-time
        sim_time += sim_dt
        count += 1

    # close the environment
    env.close()
    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()
