from __future__ import annotations


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    Articulation,
    RigidObject,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time
import numpy as np
from collections.abc import Sequence
import gymnasium as gym
import wandb

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
)
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.sensors.surface_reconstructor import SurfaceReconstructor
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTMotionGenerator,
    GTDiscreteMotionGenerator,
)

import cProfile
from gymnasium.spaces import Dict
import os

scene_cfg = YAML().load(
    open(
        f"{PACKAGE_DIR}/tasks/robot_US_reconstruction/cfgs/robotic_US_reconstruction.yaml",
        "r",
    )
)

# robot
robot_cfg = scene_cfg["robot"]
INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_cfg["joint_pos"][0],
        "lbr_joint_1": robot_cfg["joint_pos"][1],
        "lbr_joint_2": robot_cfg["joint_pos"][2],
        "lbr_joint_3": robot_cfg["joint_pos"][3],  # -1.2,
        "lbr_joint_4": robot_cfg["joint_pos"][4],
        "lbr_joint_5": robot_cfg["joint_pos"][5],  # 1.5,
        "lbr_joint_6": robot_cfg["joint_pos"][6],
    },
    pos=(
        float(robot_cfg["pos"][0]),
        float(robot_cfg["pos"][1]),
        float(robot_cfg["pos"][2]),
    ),  # ((0.0, -0.75, 0.4))
)

# patient
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(patient_cfg["pos"][0]),
        float(patient_cfg["pos"][1]),
        float(patient_cfg["pos"][2]),
    ),  # 0.7
    rot=(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])),
)

# bed
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=(
        float(bed_cfg["pos"][0]),
        float(bed_cfg["pos"][1]),
        float(bed_cfg["pos"][2]),
    ),  # 0.7
    rot=(0.5, 0.5, 0.5, 0.5),
)
scale_bed = bed_cfg["scale"]
# use stl: selected_dataset_body_contact
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_raw_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id
    for p_id in patient_cfg["id_list"]
]

target_anatomy = scene_cfg["reconstruction"]["target_vertebra"]
target_stl_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + str(target_anatomy)
    + ".stl"
    for p_id in patient_cfg["id_list"]
]
target_traj_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + "standard_right_traj_"
    + str(target_anatomy)[-2:]
    + ".stl"
    for p_id in patient_cfg["id_list"]
]

usd_file_list = [
    human_file + "/combined_wrapwrap/combined_wrapwrap.usd"
    for human_file in human_usd_list
]
label_map_file_list = [
    human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list
]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]
scale = 1 / label_res


@configclass
class roboticUSRecEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]  # 300
    action_scale = 1
    action_space = 4
    observation_space = list(scene_cfg["reconstruction"]["volume_size"])  # (40, 40, 40)
    state_space = 0
    # observation_scale = scene_cfg['observation']['scale']

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120, render_interval=decimation
    )

    robot_cfg: ArticulationCfg = KUKA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_US", init_state=INIT_STATE_ROBOT_US
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=2.0, replicate_physics=False
    )


class roboticUSRecEnv(DirectRLEnv):
    cfg: roboticUSRecEnvCfg

    def __init__(
        self, cfg: roboticUSRecEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_entity_cfg = SceneEntityCfg(
            "robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"]
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        # define ik controllers
        ik_params = {"lambda_val": 0.00001}
        pose_diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.pose_diff_ik_controller = DifferentialIKController(
            pose_diff_ik_cfg, self.scene.num_envs, device=self.sim.device
        )

        # load label maps
        label_map_list = []
        for label_map_file in label_map_file_list:
            label_map = nib.load(label_map_file).get_fdata()
            label_map_list.append(label_map)
        # load ct maps
        ct_map_list = []
        for ct_map_file in ct_map_file_list:
            ct_map = nib.load(ct_map_file).get_fdata()
            ct_min_max = scene_cfg["sim"]["ct_range"]
            ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
            ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
            ct_map_list.append(ct_map)

        # construct label image slicer
        label_convert_map = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
        )

        # construct US simulator
        self.sim_cfg = scene_cfg["sim"]

        self.vertebra_viewer = VertebraViewer(
            self.scene.num_envs,
            len(human_usd_list),
            target_stl_file_list,
            target_traj_file_list,
            self.sim_cfg["vis_us"],
            label_res,
            self.sim.device,
        )

        # construct surface reconstructor
        self.surface_reconstructor = SurfaceReconstructor(
            label_map_list,
            ct_map_list,
            human_stl_list,
            self.scene.num_envs,
            self.sim_cfg["patient_xz_range"],
            self.sim_cfg["patient_xz_init_range"][0],
            self.sim.device,
            label_convert_map,
            scene_cfg["reconstruction"]["volume_size"],
            scene_cfg["reconstruction"]["volume_res"],
            self.vertebra_viewer.human_to_ver_per_envs_real,
            target_vertebra_label=32
            - int(scene_cfg["reconstruction"]["target_vertebra"][-1]),
            missing_rate=scene_cfg["reconstruction"]["missing_rate"],
            add_height=scene_cfg["reconstruction"]["add_height"],
            target_vertebra_points=self.vertebra_viewer.vertebra_points_list,
            max_roll_adj=scene_cfg["reconstruction"]["max_roll_adj"],
            img_size=scene_cfg["reconstruction"]["img_size"],
            img_res=scene_cfg["reconstruction"]["img_res"],
            img_thickness=1,
            visualize=self.sim_cfg["vis_rec"],
        )
        self.surface_reconstructor.current_x_z_x_angle_cmd = torch.zeros(
            (3,), device=self.sim.device
        )

        self.human_world_poses = (
            self.human.data.root_state_w
        )  # these are already the initial poses

        # construct ground truth motion generator
        self.max_action = torch.tensor(
            scene_cfg["action"]["max_action"], device=self.sim.device
        ).reshape((1, -1))

        # change observation space to image
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.cfg.observation_space,
            dtype=np.uint8,
        )

        # self.cfg.observation_space[0] = self.surface_reconstructor.img_thickness

        self.single_observation_space["policy"] = Dict(
            {
                "reconstruction": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cfg.observation_space[0],
                        self.cfg.observation_space[1],
                        self.cfg.observation_space[2],
                    ),
                    dtype=np.uint8,
                ),
                "cur_length": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.termination_direct = True

        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        # reward
        self.rew_cfg = scene_cfg["reward"]

    def get_US_init_pose(self):
        # compute position change
        vertebra_to_US_2d_pos = torch.tensor(
            scene_cfg["motion_planning"]["vertebra_to_US_2d_pos"]
        ).to(self.sim.device)
        rand_disturbance = (
            torch.rand((self.scene.num_envs, 2), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["vertebra_to_US_rand_max"]
            - scene_cfg["motion_planning"]["vertebra_to_US_rand_max"]
        )
        vertebra_2d_pos = self.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        US_target_2d_pos = (
            vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0) + rand_disturbance
        )

        # change roll adj
        rand_dist_angle = (
            torch.rand((self.scene.num_envs, 1), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["US_roll_rand_max"]
            - scene_cfg["motion_planning"]["US_roll_rand_max"]
        )
        self.surface_reconstructor.roll_adj = (
            scene_cfg["motion_planning"]["US_roll_adj"]
            * torch.ones_like(vertebra_2d_pos[:, 0:1])
            + rand_dist_angle
        )

        rand_dist_angle = (
            torch.rand((self.scene.num_envs, 1), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["US_roll_rand_max"]
            - scene_cfg["motion_planning"]["US_roll_rand_max"]
        )
        US_target_2d_angle = (
            1.57 * torch.ones_like(vertebra_2d_pos[:, 0:1]) + rand_dist_angle
        )

        US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)

        self.surface_reconstructor.current_x_z_x_angle_cmd = US_target_2d
        world_to_ee_init_pos, world_to_ee_init_rot = (
            self.surface_reconstructor.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )

    def _setup_scene(self):
        """Configuration for a cart-pole scene."""

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # articulation
        # kuka US
        self.robot = Articulation(self.cfg.robot_cfg)

        # medical bad
        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_no_contact/hospital_bed.usd",
                scale=(scale_bed, scale_bed, scale_bed),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),  # Improves a lot of time count=8 0.014-0.013
            ),
            init_state=INIT_STATE_BED,
        )
        medical_bed = RigidObject(medical_bed_cfg)

        # human:
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=INIT_STATE_HUMAN,
        )
        self.human = RigidObject(human_cfg)
        # assign members
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human

    def get_action_length(self, actions: torch.Tensor):
        # compute position and rotation
        pos_l = torch.linalg.norm(actions[:, 0:2], dim=-1)
        rot_l = torch.linalg.norm(actions[:, 2:], dim=-1)

        total_l = (
            self.rew_cfg["action_length"]["w_pos"] * pos_l
            + self.rew_cfg["action_length"]["w_angle"] * rot_l
        )

        return total_l, pos_l, rot_l

    def get_traj_length(self, last_cmd_pose: torch.Tensor, cur_cmd_pose: torch.Tensor):
        # compute position and rotation
        pos_l = torch.linalg.norm(cur_cmd_pose[:, 0:2] - last_cmd_pose[:, 0:2], dim=-1)
        rot_l = torch.linalg.norm(cur_cmd_pose[:, 2:] - last_cmd_pose[:, 2:], dim=-1)

        total_l = (
            self.rew_cfg["action_length"]["w_pos"] * pos_l
            + self.rew_cfg["action_length"]["w_angle"] * rot_l
        )

        return total_l, pos_l, rot_l

    def _get_observations(self) -> dict:
        # get human frame
        self.human_world_poses = self.human.data.body_link_state_w[
            :, 0, 0:7
        ]  # these are already the initial poses
        # define world to human poses
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )
        # get ee pose w
        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        # new reconstruction
        self.surface_reconstructor.obtain_new_reconstruction_from_ee_pose(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

        observations = {
            "policy": {
                "reconstruction": self.surface_reconstructor.US_rec_volume,
                "cur_length": self.total_length.reshape((-1, 1)),
            }
        }

        if scene_cfg["sim"]["vis_us"]:
            # get human frame
            self.surface_reconstructor.visualize("seg")
            self.vertebra_viewer.update_tip_vis(
                self.human_to_ee_pos, self.human_to_ee_quat
            )

        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        action: (num_envs, 4)
        x, z, x_angle, y_angle,
        """
        # if action is 6 dim (SE), convert to 3 dim (xz pos + y rot)
        if actions.shape[-1] == 6:
            actions = actions[:, [0, 2, 4, 5]]

        # clamp action
        actions = torch.clamp(
            actions * self.action_scale, -self.max_action, self.max_action
        )
        self.actions = actions
        # update the target command
        # actions: dx, dz: in image frame
        self.human_to_ee_pos, self.human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        human_to_ee_rot_mat = matrix_from_quat(self.human_to_ee_quat)
        dx_dz_human = (
            actions[:, 0].unsqueeze(1) * human_to_ee_rot_mat[:, :, 0]
            + actions[:, 1].unsqueeze(1) * human_to_ee_rot_mat[:, :, 1]
        )
        cmd = torch.cat([dx_dz_human[:, [0, 2]], actions[:, 2:4]], dim=-1)
        self.surface_reconstructor.update_cmd(cmd)

        # compute desired world to ee pose
        world_to_ee_target_pos, world_to_ee_target_rot = (
            self.surface_reconstructor.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )
        # compute desired base to ee pose
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_rot,
        )
        base_to_ee_target_pose = torch.cat(
            [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
        )

        # set command to robot
        # set new command
        self.pose_diff_ik_controller.set_command(base_to_ee_target_pose)

    def _apply_action(self):
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        # get current ee
        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        # # get joint position targets
        US_jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
        ]
        US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = self.pose_diff_ik_controller.compute(
            self.US_ee_pos_b, self.US_ee_quat_b, US_jacobian, US_joint_pos
        )
        # apply joint oosition target
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = 0
        # length of path
        self.act_l, self.pos_l, self.rot_l = self.get_action_length(self.actions)
        cmd_state = self.surface_reconstructor.human_cmd_state_from_ee_pose(
            self.human_to_ee_pos, self.human_to_ee_quat
        )
        self.total_l, self.pos_l, self.rot_l = self.get_traj_length(
            self.cmd_state, cmd_state
        )

        # current coverage
        reward += (
            self.rew_cfg["incremental_cov"] * self.surface_reconstructor.incremental_cov
        )
        reward -= self.total_l

        self.total_length += self.act_l
        self.total_pos_l += self.pos_l
        self.total_rot_l += self.rot_l
        self.total_reward += reward
        self.cov_ratio = self.surface_reconstructor.get_converage_ratio()
        self.cmd_state = cmd_state
        # print(self.total_reward)
        # print(self.surface_reconstructor.cur_cov)
        # print(self.total_length)
        # record infor
        self.extras["human_to_ee_pos"] = self.human_to_ee_pos
        self.extras["human_to_ee_quat"] = self.human_to_ee_quat
        self.extras["cur_cmd_state"] = self.cmd_state

        if scene_cfg["if_record_traj"]:
            self.cmd_pose_trajs.append(self.cmd_state)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.termination_direct:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(self.episode_length_buf)
        out_of_bounds = torch.zeros_like(
            self.surface_reconstructor.no_collide
        )  # self.US_slicer.no_collide

        return out_of_bounds, time_out

    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 200,
    ):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(num_steps):
            self._sim_step_counter += 1
            # set actions into buffers

            # get human frame
            self.human_world_poses = self.human.data.body_link_state_w[
                :, 0, 0:7
            ]  # these are already the initial poses
            # define world to human poses
            self.world_to_human_pos, self.world_to_human_rot = (
                self.human_world_poses[:, 0:3],
                self.human_world_poses[:, 3:7],
            )
            world_ee_target_pos, world_ee_target_quat = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # get current joint positions
            self.US_ee_pose_w = self.robot.data.body_state_w[
                :, self.robot_entity_cfg.body_ids[-1], 0:7
            ]

            # get current ee
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                world_ee_target_pos,
                world_ee_target_quat,
            )
            base_to_ee_target_pose = torch.cat(
                [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
            )

            # set new command
            self.pose_diff_ik_controller.set_command(base_to_ee_target_pose)

            # get joint position targets
            US_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
            ]
            US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            # compute the joint commands
            joint_pos_des = self.pose_diff_ik_controller.compute(
                US_ee_pos_b, US_ee_quat_b, US_jacobian, US_joint_pos
            )
            self.robot.set_joint_position_target(
                joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
            )

            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        self.pose_diff_ik_controller.reset()

        # get ee pose in base frame
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]

        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]
        # compute frame in root frame
        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            self.US_root_pose_w[:, 0:3],
            self.US_root_pose_w[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.pose_diff_ik_controller.action_dim,
            device=self.sim.device,
        )
        self.pose_diff_ik_controller.set_command(
            ik_commands_pose, self.US_ee_pos_b, self.US_ee_quat_b
        )

        # inverse kinematics?
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        # get human frame
        self.human_world_poses = self.human.data.body_link_state_w[
            :, 0, 0:7
        ]  # these are already the initial poses
        # define world to human poses
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        # compute 2d target poses
        self.get_US_init_pose()
        # compute joint positions
        # set joint positions
        self._move_towards_target(
            self.surface_reconstructor.human_to_ee_target_pos,
            self.surface_reconstructor.human_to_ee_target_quat,
        )

        # actions: dx, dz: in image frame
        self.human_to_ee_pos, self.human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        self.cmd_state = self.surface_reconstructor.human_cmd_state_from_ee_pose(
            self.human_to_ee_pos, self.human_to_ee_quat
        )
        if hasattr(self, "total_reward"):
            wandb.log({"total_reward": self.total_reward.mean().item()})
            wandb.log({"total_length": self.total_length.mean().item()})
            wandb.log({"cov_ratio": self.cov_ratio.mean().item()})
            wandb.log({"total_pos_l": (self.total_pos_l).mean().item()})
            wandb.log({"total_rot_l": (self.total_rot_l * 180 / torch.pi).mean().item()})
            wandb.log({"cov_ratio std": self.cov_ratio.std().item()})
            wandb.log({"total_pos_l std": self.total_pos_l.std().item()})
            wandb.log({"total_rot_l std": (self.total_rot_l * 180 / torch.pi).std().item()})
        self.total_reward = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_length = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_rot_l = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_pos_l = torch.zeros(self.scene.num_envs, device=self.sim.device)

        # reset the surface reconstructor
        self.surface_reconstructor.reset()

        # record infor
        self.extras["human_to_ee_pos"] = self.human_to_ee_pos
        self.extras["human_to_ee_quat"] = self.human_to_ee_quat
        self.extras["cur_cmd_state"] = self.cmd_state

        if scene_cfg["if_record_traj"]:
            record_path = PACKAGE_DIR + scene_cfg["record_path"]
            if hasattr(self, "cmd_pose_trajs"):
                if not os.path.exists(record_path):
                    os.makedirs(record_path)
                self.cmd_pose_trajs = torch.stack(self.cmd_pose_trajs, dim=1)
                torch.save(self.cmd_pose_trajs, record_path + "cmd_pose_trajs.pt")
                torch.save(
                    self.vertebra_viewer.human_to_ver_per_envs,
                    record_path + "human_to_ver.pt",
                )
            self.cmd_pose_trajs = [self.cmd_state]

    def check_nan(self):
        if torch.isnan(self.US_ee_pos_b).any() or torch.isnan(self.US_ee_quat_b).any():
            print("US_ee_pos_b", self.US_ee_pos_b)
            print("US_ee_quat_b", self.US_ee_quat_b)
            raise ValueError("nan value detected")
        if torch.isnan(self.surface_reconstructor.incremental_cov).any():
            print("incremental cov", self.surface_reconstructor.incremental_cov)
            raise ValueError("nan value detected")
        if torch.isnan(self.surface_reconstructor.cur_cov).any():
            print("cur cov", self.surface_reconstructor.cur_cov)
            raise ValueError("nan value detected")
