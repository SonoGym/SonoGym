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

from config import EvalPipelineConfig


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    ##
    # Launch Isaac Sim Simulator first.
    ##

    from isaaclab.app import AppLauncher

    # launch the simulator
    app_launcher = AppLauncher()
    simulation_app = app_launcher.app

    ##
    # Rest everything follows.
    ##

    from isaaclab.envs import ManagerBasedEnv

    from isaac.locoma.tasks.locomanipulation.door_manipulate.door_manipulate_mock_env_cfg import (
        DoorManipulationMockEnvCfg,
    )

    # Prepare the environment configuration
    env_cfg = DoorManipulationMockEnvCfg()
    env_cfg.scene.num_envs = cfg.eval.batch_size
    env_cfg.seed = cfg.seed

    # Create the environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Load the dataset
    dataset_meta = LeRobotDatasetMetadata("rsl/horizontal-door-opening")
    # Create the policy
    policy: PreTrainedPolicy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
    policy.eval()

    # Define simulation stepping
    sim_dt = env.step_dt
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset all the environments
            obs, _ = env.reset()

            # rename keys of the dictionary to match lerobot
            obs = {
                "observation.state": obs["state"],
                "observation.environment_state": obs["environment"],
            }

            # reset the policy
            policy.reset()

        # select action
        with torch.inference_mode():
            action = policy.select_action(obs)

        # step the environment
        obs, _ = env.step(action)
        # rename keys of the dictionary to match lerobot
        obs = {
            "observation.state": obs["state"],
            "observation.environment_state": obs["environment"],
        }

        # update sim-time
        sim_time += sim_dt
        count += 1

    # close the environment
    env.close()
    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()
