# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guidance-v0', help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import spinal_surgery
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.agents.torch.sac import SAC, SAC_CFG
from spinal_surgery.lab.agents.skrl_actor_critic import SharedModelSAC
import wandb
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
agent_cfg_entry_point = "skrl_sac_cfg_entry_point"


def _resolve_agent_section(cfg: dict) -> dict:
    """Return the flat SAC-config section.

    Some yaml configs are nested (skrl Runner format with ``agent:`` subkey),
    others are flat. Pick the inner ``agent`` block when present.
    """
    return cfg.get("agent", cfg)


def _normalize_sac_cfg(cfg: dict) -> dict:
    """Map legacy yaml keys to the new SAC_CFG fields and drop unknown keys."""
    cfg = dict(cfg)
    # the script has a separate "rollouts" config used for memory sizing only.
    cfg.pop("rollouts", None)
    cfg.pop("class", None)
    cfg.pop("seed", None)

    # Compose ``learning_rate`` tuple if old per-network rates are provided.
    actor_lr = cfg.pop("actor_learning_rate", None)
    critic_lr = cfg.pop("critic_learning_rate", None)
    entropy_lr = cfg.pop("entropy_learning_rate", None)
    if any(v is not None for v in (actor_lr, critic_lr, entropy_lr)):
        base_lr = cfg.get("learning_rate", 1e-3)
        if not isinstance(base_lr, (tuple, list)):
            base_lr = (base_lr, base_lr, base_lr)
        cfg["learning_rate"] = (
            actor_lr if actor_lr is not None else base_lr[0],
            critic_lr if critic_lr is not None else base_lr[1],
            entropy_lr if entropy_lr is not None else base_lr[2],
        )

    valid_keys = {f.name for f in dataclasses.fields(SAC_CFG)}
    return {k: v for k, v in cfg.items() if k in valid_keys}


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    sac_section = _resolve_agent_section(agent_cfg)
    sac_section["seed"] = args_cli.seed if args_cli.seed is not None else sac_section.get("seed")
    env_cfg.seed = sac_section["seed"]

    # max iterations for training
    trainer_section = agent_cfg.get("trainer", {})
    if args_cli.max_iterations:
        trainer_section["timesteps"] = args_cli.max_iterations * sac_section.get("rollouts", 1)

    # specify directory for logging experiments
    experiment_section = sac_section.setdefault("experiment", {})
    log_root_path = os.path.join("logs", "skrl", experiment_section.get("directory", ""))
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_sac_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if experiment_section.get("experiment_name"):
        log_dir += f'_{experiment_section["experiment_name"]}'
    experiment_section["directory"] = log_root_path
    experiment_section["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # init wandb
    wandb.init(project=args_cli.task, config=agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    models = {}
    device = env_cfg.sim.device
    models['policy'] = SharedModelSAC(env.observation_space, env.action_space, device, clip_actions=False)
    models["critic_1"] = models["policy"]
    models["critic_2"] = models["policy"]
    models["target_critic_1"] = SharedModelSAC(env.observation_space, env.action_space, device, clip_actions=False)
    models["target_critic_2"] = SharedModelSAC(env.observation_space, env.action_space, device, clip_actions=False)

    # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory_size = sac_section.get("rollouts", 64)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device, replacement=False)

    sac_cfg_dict = _normalize_sac_cfg(sac_section)
    # ensure the experiment block is preserved through normalization
    sac_cfg_dict["experiment"] = experiment_section

    sac_agent = SAC(
        models=models,
        memory=memory,
        cfg=sac_cfg_dict,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Configure and instantiate the RL trainer
    trainer_timesteps = trainer_section.get("timesteps", 200000)
    cfg_trainer = SequentialTrainerCfg(timesteps=trainer_timesteps, headless=True)
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=sac_agent)

    # start training
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
