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

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit_assets import ASSETS_DATA_DIR

##
# Configuration
##

KUKA_DRILL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/Robots/Kuka/med14/kuka_drill.usd", # transparent
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
            "lbr_joint_0": 0.0,
            "lbr_joint_1": 0.0,
            "lbr_joint_2": 0.0,
            "lbr_joint_3": -1.2,
            "lbr_joint_4": 0.0,
            "lbr_joint_5": 1.5,
            "lbr_joint_6": 0.0,
        },
    ),
    actuators={
        "kuka_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["lbr_joint_[0-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "kuka_forearm": ImplicitActuatorCfg(
            joint_names_expr=["lbr_joint_[5-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


KUKA_HIGH_PD_DRILL_CFG = KUKA_DRILL_CFG.copy()
KUKA_HIGH_PD_DRILL_CFG.spawn.rigid_props.disable_gravity = True
KUKA_HIGH_PD_DRILL_CFG.actuators["kuka_shoulder"].stiffness = 400.0
KUKA_HIGH_PD_DRILL_CFG.actuators["kuka_shoulder"].damping = 80.0
KUKA_HIGH_PD_DRILL_CFG.actuators["kuka_forearm"].stiffness = 400.0
KUKA_HIGH_PD_DRILL_CFG.actuators["kuka_forearm"].damping = 80.0
"""Configuration of kuka robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
