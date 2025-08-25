# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from spinal_surgery import ASSETS_DATA_DIR

##
# Configuration
##

FR3_US_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/Robots/Franka/fr3_US.usd",  # transparent
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": -1.2,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.5,
            "fr3_joint7": 0.0,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[6-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FR3_HIGH_PD_US_CFG = FR3_US_CFG.copy()
FR3_HIGH_PD_US_CFG.spawn.rigid_props.disable_gravity = True
FR3_HIGH_PD_US_CFG.actuators["fr3_shoulder"].stiffness = 400.0
FR3_HIGH_PD_US_CFG.actuators["fr3_shoulder"].damping = 80.0
FR3_HIGH_PD_US_CFG.actuators["fr3_forearm"].stiffness = 400.0
FR3_HIGH_PD_US_CFG.actuators["fr3_forearm"].damping = 80.0
"""Configuration of fr3 robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
