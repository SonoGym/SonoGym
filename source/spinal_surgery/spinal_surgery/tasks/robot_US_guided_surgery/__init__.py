# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents


##
# Register Gym environments.
##

gym.register(
    id="Isaac-robot-US-guided-surgery-v0",
    entry_point=f"{__name__}.robotic_US_guided_surgery:roboticUSGuidedSurgeryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotic_US_guided_surgery:roboticUSGuidedSurgeryCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_ppol_cfg_entry_point": f"{agents.__name__}:skrl_ppol_cfg.yaml",
        "skrl_sppo_cfg_entry_point": f"{agents.__name__}:skrl_sppo_cfg.yaml",
        "skrl_td3_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
        "skrl_a2c_cfg_entry_point": f"{agents.__name__}:skrl_a2c_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "fsrl_cfg_entry_point": f"{agents.__name__}:fsrl_ppol_cfg.yaml",
        "fsrl_cpo_cfg_entry_point": f"{agents.__name__}:fsrl_cpo_cfg.yaml",
    },
)