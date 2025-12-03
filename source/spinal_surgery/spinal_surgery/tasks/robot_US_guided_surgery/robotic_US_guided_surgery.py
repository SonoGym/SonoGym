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
import pyvista as pv
import copy

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from spinal_surgery.assets.fr3_US import *
from spinal_surgery.assets.kuka_drill import *
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
)
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, apply_delta_pose
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTMotionGenerator,
    GTDiscreteMotionGenerator,
)
import cProfile
from gymnasium.spaces import Dict
import wandb

scene_cfg = YAML().load(
    open(
        f"{PACKAGE_DIR}/tasks/robot_US_guided_surgery/cfgs/robotic_US_guided_surgery.yaml",
        "r",
    )
)
# TODO: fix observation scale
if scene_cfg["sim"]["us"] == "net":
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]
us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
us_generative_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
)
robot_cfg = scene_cfg["robot"]

# robot
if scene_cfg["robot"]["type"] == "kuka":
    robot_articulation_cfg = KUKA_HIGH_PD_CFG
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

elif scene_cfg["robot"]["type"] == "fr3":
    robot_articulation_cfg = FR3_HIGH_PD_US_CFG
    INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": robot_cfg["joint_pos"][0],
            "fr3_joint2": robot_cfg["joint_pos"][1],
            "fr3_joint3": robot_cfg["joint_pos"][2],
            "fr3_joint4": robot_cfg["joint_pos"][3],  # -1.2,
            "fr3_joint5": robot_cfg["joint_pos"][4],
            "fr3_joint6": robot_cfg["joint_pos"][5],  # 1.5,
            "fr3_joint7": robot_cfg["joint_pos"][6],
        },
        pos=(
            float(robot_cfg["pos"][0]),
            float(robot_cfg["pos"][1]),
            float(robot_cfg["pos"][2]),
        ),  # ((0.0, -0.75, 0.4))
    )

robot_drill_cfg = scene_cfg["robot_drill"]
INIT_STATE_ROBOT_DRILL = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_drill_cfg["joint_pos"][0],
        "lbr_joint_1": robot_drill_cfg["joint_pos"][1],
        "lbr_joint_2": robot_drill_cfg["joint_pos"][2],
        "lbr_joint_3": robot_drill_cfg["joint_pos"][3],  # -1.2,
        "lbr_joint_4": robot_drill_cfg["joint_pos"][4],
        "lbr_joint_5": robot_drill_cfg["joint_pos"][5],  # 1.5,
        "lbr_joint_6": robot_drill_cfg["joint_pos"][6],
    },
    pos=(
        float(robot_drill_cfg["pos"][0]),
        float(robot_drill_cfg["pos"][1]),
        float(robot_drill_cfg["pos"][2]),
    ),  # ((0.0, -0.75, 0.4))
)
DRILL_TO_TIP_POS = np.array([0.0, 0.0, -0.135]).astype(np.float32)  # -0.135
DRILL_TO_TIP_QUAT = (
    R.from_euler("YXZ", [180, 0, 0], degrees=True).as_quat().astype(np.float32)
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
# use stl: Totalsegmentator_dataset_v2_subset_body_contact
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
target_anatomy = patient_cfg["target_anatomy"]
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
class roboticUSGuidedSurgeryCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]  # 5 # 300
    action_scale = 1
    action_space = 6
    observation_space = [
        us_cfg["image_3D_thickness"] // scene_cfg["observation"]["downsample"],
        200 // scene_cfg["observation"]["downsample"],
        150 // scene_cfg["observation"]["downsample"],
    ]
    state_space = 0
    observation_scale = scene_cfg["observation"]["scale"]

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120, render_interval=decimation
    )

    robot_cfg: ArticulationCfg = robot_articulation_cfg.replace(
        prim_path="/World/envs/env_.*/Robot_US", init_state=INIT_STATE_ROBOT_US
    )

    robot_drill_cfg: ArticulationCfg = KUKA_HIGH_PD_DRILL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_drill", init_state=INIT_STATE_ROBOT_DRILL
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=2.0, replicate_physics=False
    )


class roboticUSGuidedSurgeryEnv(DirectRLEnv):
    cfg: roboticUSGuidedSurgeryCfg

    def __init__(
        self, cfg: roboticUSGuidedSurgeryCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if scene_cfg["robot"]["type"] == "kuka":
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"]
            )
        else:
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["fr3_joint.*"], body_names=["fr3_link8"]
            )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        self.robot_drill_entity_cfg = SceneEntityCfg(
            "robot_drill", joint_names=["lbr_joint_.*"], body_names=["screw_6_65"]
        )  # "screw_6_65"
        self.robot_drill_entity_cfg.resolve(self.scene)
        self.drill_ee_jacobi_idx = self.robot_drill_entity_cfg.body_ids[-1]

        # define ik controllers
        ik_params = {"lambda_val": 0.01}
        pose_diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.pose_diff_ik_controller = DifferentialIKController(
            pose_diff_ik_cfg, self.scene.num_envs, device=self.sim.device
        )
        diff_ik_cfg_drill = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.diff_ik_controller_drill = DifferentialIKController(
            diff_ik_cfg_drill, self.scene.num_envs, device=self.sim.device
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
        self.init_cmd_pose_min = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][0], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.init_cmd_pose_max = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][1], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        if scene_cfg["observation"]["3D"]:
            img_thickness = us_cfg["image_3D_thickness"]
        else:
            img_thickness = 1

        # down sample
        res = scene_cfg["observation"]["downsample"]
        us_cfg["image_size"] = [
            int(us_cfg["image_size"][0] / res),
            int(us_cfg["image_size"][1] / res),
        ]
        us_cfg["system_params"]["sx_E"] = us_cfg["system_params"]["sx_E"] / np.sqrt(res)
        us_cfg["system_params"]["sy_E"] = us_cfg["system_params"]["sy_E"] / np.sqrt(res)
        us_cfg["system_params"]["sx_B"] = us_cfg["system_params"]["sx_B"] / np.sqrt(res)
        us_cfg["system_params"]["sy_B"] = us_cfg["system_params"]["sy_B"] / np.sqrt(res)
        us_cfg["system_params"]["I0"] *= np.sqrt(res)
        us_cfg["E_S_ratio"] /= np.sqrt(res)
        img_thickness = max(int(img_thickness // res), 1)
        us_cfg["resolution"] = us_cfg["resolution"] * res

        self.US_slicer = USSlicer(
            us_cfg,
            label_map_list,
            ct_map_list,
            self.sim_cfg["if_use_ct"],
            human_stl_list,
            self.scene.num_envs,
            self.sim_cfg["patient_xz_range"],
            self.sim_cfg["patient_xz_init_range"][0],
            self.sim.device,
            label_convert_map,
            us_cfg["image_size"],
            us_cfg["resolution"],
            img_thickness=img_thickness,
            roll_adj=scene_cfg["motion_planning"]["US_roll_adj"],
            visualize=self.sim_cfg["vis_seg_map"],
            sim_mode=scene_cfg["sim"]["us"],
            us_generative_cfg=us_generative_cfg,
        )
        self.US_slicer.current_x_z_x_angle_cmd = (
            self.init_cmd_pose_min + self.init_cmd_pose_max
        ) / 2

        self.human_world_poses = (
            self.human.data.root_state_w
        )  # these are already the initial poses

        # construct ground truth motion generator
        motion_plan_cfg = scene_cfg["motion_planning"]
        self.vertebra_viewer = VertebraViewer(
            self.scene.num_envs,
            len(human_usd_list),
            target_stl_file_list,
            target_traj_file_list,
            self.sim_cfg["vis_us"],
            label_res,
            self.sim.device,
        )

        # change observation space to image
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.cfg.observation_space[0], self.cfg.observation_space[1], self.cfg.observation_space[2]),
        #     dtype=np.uint8,
        # )
        # drill rand
        self.rand_joint_pos_max = (
            torch.tensor(motion_plan_cfg["joint_pos_rand_max"])
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
            .to(self.sim.device)
        )

        self.cfg.observation_space[0] = self.US_slicer.img_thickness
        if scene_cfg["sim"]["us"] == "net":
            self.cfg.observation_space[0] = (
                self.cfg.observation_space[0]
                // us_generative_cfg["elevation_downsample"]
            )

        self.single_observation_space["policy"] = Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cfg.observation_space[0],
                        self.cfg.observation_space[1],
                        self.cfg.observation_space[2],
                    ),
                    dtype=np.uint8,
                ),
                "pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "quat": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        self.observation_space = Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.scene.num_envs,
                        self.cfg.observation_space[0],
                        self.cfg.observation_space[1],
                        self.cfg.observation_space[2],
                    ),
                    dtype=np.uint8,
                ),
                "pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.scene.num_envs, 3),
                    dtype=np.float32,
                ),
                "quat": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.scene.num_envs, 4),
                    dtype=np.float32,
                ),
            }
        )

        self.termination_direct = True
        self.observation_mode = scene_cfg["observation"]["mode"]
        self.action_mode = scene_cfg["action"]["mode"]
        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        # for reward
        self.safe_height = scene_cfg["reward"]["safe_height"]
        self.w_pos = scene_cfg["reward"]["w_pos"]
        self.w_angle = scene_cfg["reward"]["w_angle"]
        self.w_cost = scene_cfg["reward"]["w_cost"]
        self.w_insertion = scene_cfg["reward"]["w_insertion"]

        # action scale
        self.max_action = (
            torch.tensor(scene_cfg["action"]["max_action"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.min_action = (
            -torch.tensor(scene_cfg["action"]["max_action"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        # discrete action
        if scene_cfg["action"]["mode"] == "discrete":
            self.single_action_space = gym.spaces.Discrete(
                self.cfg.action_space * 2 + 1
            )
        else:
            self.single_action_space = gym.spaces.Box(
                low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
                high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
                shape=(self.cfg.action_space,),
                dtype=np.float32,
            )

        # wandb.init()
        self.num_step = 0

    def get_US_target_pose(self):
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
        self.US_slicer.roll_adj = (
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

        self.US_slicer.current_x_z_x_angle_cmd = US_target_2d
        world_to_ee_init_pos, world_to_ee_init_rot = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
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

        self.robot_drill = Articulation(self.cfg.robot_drill_cfg)

        # medical bad
        if scene_cfg["sim"]["vis_us"]:
            usd_folder = "usd_colored"
        else:
            usd_folder = "usd_no_contact"
        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/"
                + usd_folder
                + "/hospital_bed.usd",
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
        self.scene.articulations["robot_drill"] = self.robot_drill
        self.scene.rigid_objects["human"] = self.human

        self.drill_to_tip_pos = (
            torch.tensor(DRILL_TO_TIP_POS, device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.drill_to_tip_quat = (
            torch.tensor(DRILL_TO_TIP_QUAT, device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.tip_to_drill_pos, self.tip_to_drill_quat = subtract_frame_transforms(
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
            torch.zeros_like(self.drill_to_tip_pos).to(self.sim.device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            .to(self.sim.device)
            .repeat(self.scene.num_envs, 1),
        )

    def action_discrete_to_continuous(self, action):
        # action = 0, 1, 2,...,12
        cont_actions = torch.zeros(
            (self.scene.num_envs, self.cfg.action_space), device=self.sim.device
        )
        total_inds = torch.arange(self.scene.num_envs, device=self.sim.device)

        non_zero_inds = total_inds[action.reshape((-1,)) != 12]  # (K_n,)
        non_zero_dim = (action[non_zero_inds] // 2).to(
            torch.int
        )  # 0, 1, 2, 3, 4, 5 (K_n,)

        action_scale = self.max_action[non_zero_inds, :]  # (k_n, 6)
        action_scale = (
            action_scale[
                torch.arange(action_scale.shape[0]).to(self.sim.device), non_zero_dim
            ]
            / 2
        )  # (k_n,)
        action_sign = (action[non_zero_inds] % 2) * 2 - 1  # (k_n, 6)
        cont_actions[non_zero_inds, non_zero_dim] = action_scale * action_sign
        return cont_actions

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
        self.num_step += 1

        if self.observation_mode == "US":
            self.US_slicer.slice_US(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.us_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
        elif self.observation_mode == "CT":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.ct_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )

        elif self.observation_mode == "seg":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.label_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
        else:
            raise ValueError("Invalid observation mode")

        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize(self.observation_mode)

        # get drill to US pose
        self.US_to_drill_pos, self.US_to_drill_quat = subtract_frame_transforms(
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )
        self.US_to_tip_pos, self.US_to_tip_quat = combine_frame_transforms(
            self.US_to_drill_pos,
            self.US_to_drill_quat,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )

        observations = {
            "policy": {
                "image": obs_img,
                "pos": self.US_to_drill_pos,
                "quat": self.US_to_drill_quat,
            }
        }

        self.check_nan()

        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # control drill robot
        self.get_drill_ee_pose_b()

        if self.action_mode == "discrete":
            actions = self.action_discrete_to_continuous(actions)
        else:
            actions = actions * self.action_scale
            actions = torch.clamp(actions, self.min_action, self.max_action)

        # apply physics constraints
        self.get_traj_to_tip_state()

        # limit range
        too_high = self.tip_pos_along_traj < -0.5
        actions[too_high, :] = 0

        safety_critical = self.tip_pos_along_traj > -self.safe_height
        safe_close = torch.logical_and(
            safety_critical,
            self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002,
        )
        safe_close = torch.logical_and(
            safe_close, self.tip_pos_along_traj < self.vertebra_viewer.traj_half_length
        )
        actions[safe_close, 0:2] *= 0.2
        actions[safe_close, 3:] *= 0.2

        # action in ee space
        tip_to_next_tip_pos, tip_to_next_tip_quat = apply_delta_pose(
            torch.zeros_like(self.drill_ee_pos_b).to(self.scene.device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            .to(self.scene.device)
            .repeat(self.scene.num_envs, 1),
            actions,
        )
        tip_pos_b, tip_quat_b = combine_frame_transforms(
            self.drill_ee_pos_b,
            self.drill_ee_quat_b,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )
        next_tip_pos_b, next_tip_quat_b = combine_frame_transforms(
            tip_pos_b, tip_quat_b, tip_to_next_tip_pos, tip_to_next_tip_quat
        )
        # next drill_pos
        drill_next_ee_pos_b, drill_next_ee_quat_b = combine_frame_transforms(
            next_tip_pos_b,
            next_tip_quat_b,
            self.tip_to_drill_pos,
            self.tip_to_drill_quat,
        )

        self.diff_ik_controller_drill.set_command(
            torch.cat([drill_next_ee_pos_b, drill_next_ee_quat_b], dim=-1)
        )

    def _apply_action(self):
        self.get_drill_ee_pose_b()
        # # get joint position targets
        drill_jacobian = self.robot_drill.root_physx_view.get_jacobians()[
            :, self.drill_ee_jacobi_idx - 1, :, self.robot_drill_entity_cfg.joint_ids
        ]
        drill_joint_pos = self.robot_drill.data.joint_pos[
            :, self.robot_drill_entity_cfg.joint_ids
        ]
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller_drill.compute(
            self.drill_ee_pos_b, self.drill_ee_quat_b, drill_jacobian, drill_joint_pos
        )
        # apply joint oosition target
        self.robot_drill.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_drill_entity_cfg.joint_ids
        )

        self._apply_us_command()

    def _apply_us_command(self):
        self.get_US_ee_pose_b()

        world_ee_target_pos, world_ee_target_quat = combine_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
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
            self.US_ee_pos_b, self.US_ee_quat_b, US_jacobian, US_joint_pos
        )
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )

    def _get_rewards(self) -> torch.Tensor:

        reward = torch.zeros(self.scene.num_envs, device=self.sim.device)
        penalty = torch.zeros(self.scene.num_envs, device=self.sim.device)
        cost = torch.zeros(self.scene.num_envs, device=self.sim.device)

        if self.sim_cfg["vis_us"]:
            self.vertebra_viewer.update_tip_vis(
                self.human_to_tip_pos, self.human_to_tip_quat
            )

        # free space
        free_region = self.tip_pos_along_traj <= -self.safe_height
        reward[free_region] += self.w_pos * (
            torch.abs(self.last_tip_pos_along_traj[free_region] + self.safe_height)
            - torch.abs(self.tip_pos_along_traj[free_region] + self.safe_height)
        )

        # safety critical space
        safety_critical = self.tip_pos_along_traj > -self.safe_height
        safe_close = torch.logical_and(
            safety_critical,
            self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002,
        )
        close = self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002
        safe_close = torch.logical_and(
            safe_close, self.tip_pos_along_traj < self.vertebra_viewer.traj_half_length
        )
        unsafe = torch.logical_and(torch.logical_not(safe_close), safety_critical)
        self.ever_unsafe[unsafe] = 1
        always_safe = torch.logical_not(self.ever_unsafe)

        # reward insertion
        always_safe_and_close = torch.logical_and(always_safe, safe_close)
        self.max_tip_pos_along_traj = torch.maximum(
            self.tip_pos_along_traj, self.max_tip_pos_along_traj
        )
        reward[safe_close] += self.w_insertion * (
            torch.abs(
                self.last_max_tip_pos_along_traj[safe_close]
                - self.vertebra_viewer.traj_half_length[safe_close]
            )
            - torch.abs(
                self.max_tip_pos_along_traj[safe_close]
                - self.vertebra_viewer.traj_half_length[safe_close]
            )
        )
        # to avoid loop
        self.total_insertion[safe_close] += torch.abs(
            self.last_max_tip_pos_along_traj[safe_close]
            - self.vertebra_viewer.traj_half_length[safe_close]
        ) - torch.abs(
            self.max_tip_pos_along_traj[safe_close]
            - self.vertebra_viewer.traj_half_length[safe_close]
        )

        # unsafe penalty
        # last_safe_now_unsafe = torch.logical_and(torch.logical_not(self.last_unsafe), unsafe)
        last_safe_close_now_unsafe = torch.logical_and(self.last_safe_close, unsafe)
        last_unsafe_now_safe_close = torch.logical_and(self.last_unsafe, safe_close)
        reward[last_unsafe_now_safe_close] += (
            self.w_cost * self.total_insertion[last_unsafe_now_safe_close]
        )

        penalty[last_safe_close_now_unsafe] = self.total_insertion[
            last_safe_close_now_unsafe
        ]

        reward += self.w_pos * (self.last_tip_to_traj_dist - self.tip_to_traj_dist)
        reward += self.w_angle * (
            torch.abs(self.last_traj_to_tip_sin) - torch.abs(self.traj_to_tip_sin)
        )
        # TODO: add more for safe region
        any_close = torch.logical_or(
            self.last_close,
            close,
        )
        reward[any_close] += self.w_insertion * (
            self.last_tip_to_traj_dist[any_close] - self.tip_to_traj_dist[any_close]
        )

        reward -= penalty * self.w_cost

        # TODO: record total distance
        self.total_dist = torch.sqrt(
            torch.abs(self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length)
            ** 2
            + self.tip_to_traj_dist
        )
        self.angle = torch.asin(self.traj_to_tip_sin) * 180 / torch.pi
        self.insert_err = torch.abs(
            self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length
        )
        self.inserted = (
            self.tip_pos_along_traj > -self.vertebra_viewer.traj_half_length
        ).reshape((-1,))

        self.last_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        self.last_tip_to_traj_dist = copy.deepcopy(self.tip_to_traj_dist)
        self.last_traj_to_tip_sin = copy.deepcopy(self.traj_to_tip_sin)
        self.last_unsafe = copy.deepcopy(unsafe)
        self.last_safe_close = copy.deepcopy(safe_close)
        self.last_close = copy.deepcopy(close)
        # self.last_traj_pos_along_traj_safe_close[safe_close] = copy.deepcopy(self.tip_pos_along_traj[safe_close])
        self.last_max_tip_pos_along_traj = copy.deepcopy(self.max_tip_pos_along_traj)

        self.total_rewards += reward

        # TODO: update cost for safe learning
        self.extras["cost"] = unsafe.to(torch.float32)
        self.total_costs += self.extras["cost"]

        # record information
        ones = torch.ones((self.scene.num_envs,), device=self.sim.device)
        self.extras["traj_drct"] = self.vertebra_viewer.traj_drct
        self.extras["human_to_tip_pos"] = self.human_to_tip_pos
        self.extras["human_to_tip_quat"] = self.human_to_tip_quat
        self.extras["safe_height"] = self.safe_height * ones
        self.extras["traj_half_length"] = self.vertebra_viewer.traj_half_length
        self.extras["traj_radius"] = self.vertebra_viewer.traj_radius
        self.extras["tip_to_traj_dist"] = self.tip_to_traj_dist
        self.extras["traj_to_tip_sin"] = self.traj_to_tip_sin
        self.extras["human_to_traj_pos"] = self.vertebra_viewer.human_to_traj_pos
        self.extras["tip_pos_along_traj"] = self.tip_pos_along_traj

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.termination_direct:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(self.episode_length_buf)
        out_of_bounds = torch.zeros_like(
            self.US_slicer.no_collide
        )  # self.US_slicer.no_collide

        return out_of_bounds, time_out

    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 300,
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

            self._apply_us_command()

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

    def get_US_ee_pose_b(self):
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

    def get_drill_ee_pose_b(self):
        self.drill_root_pose_w = self.robot_drill.data.root_state_w[:, 0:7]

        self.drill_ee_pose_w = self.robot_drill.data.body_state_w[
            :, self.robot_drill_entity_cfg.body_ids[-1], 0:7
        ]
        # compute frame in root frame
        self.drill_ee_pos_b, self.drill_ee_quat_b = subtract_frame_transforms(
            self.drill_root_pose_w[:, 0:3],
            self.drill_root_pose_w[:, 3:7],
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )

    def get_traj_to_tip_state(self):
        self.human_to_ee_pos, self.human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )
        self.human_to_tip_pos, self.human_to_tip_quat = combine_frame_transforms(
            self.human_to_ee_pos,
            self.human_to_ee_quat,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )
        self.tip_pos_along_traj, self.tip_to_traj_dist, self.traj_to_tip_sin = (
            self.vertebra_viewer.compute_tip_in_traj(
                self.human_to_tip_pos, self.human_to_tip_quat
            )
        )

    def reset_controllers(self):
        self.pose_diff_ik_controller.reset()
        self.diff_ik_controller_drill.reset()

        self.get_US_ee_pose_b()

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.pose_diff_ik_controller.action_dim,
            device=self.sim.device,
        )
        self.pose_diff_ik_controller.set_command(
            ik_commands_pose, self.US_ee_pos_b, self.US_ee_quat_b
        )

        self.get_drill_ee_pose_b()

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.diff_ik_controller_drill.action_dim,
            device=self.sim.device,
        )
        self.diff_ik_controller_drill.set_command(
            ik_commands_pose, self.drill_ee_pos_b, self.drill_ee_quat_b
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reconstruct random maps
        self.US_slicer.construct_T_maps()
        self.US_slicer.construct_Vl_maps()

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        self.robot.set_joint_position_target(
            joint_pos, joint_ids=self.robot_entity_cfg.joint_ids
        )

        joint_pos = self.robot_drill.data.default_joint_pos.clone()
        joint_vel = self.robot_drill.data.default_joint_vel.clone()

        rand_joint_pos = (
            torch.rand(
                (self.scene.num_envs, self.rand_joint_pos_max.shape[1]),
                device=self.sim.device,
            )
            * 2
            - 1
        )
        rand_joint_pos = rand_joint_pos * self.rand_joint_pos_max

        self.robot_drill.write_joint_state_to_sim(joint_pos + rand_joint_pos, joint_vel)
        self.robot_drill.reset()
        self.robot_drill.set_joint_position_target(
            joint_pos + rand_joint_pos, joint_ids=self.robot_drill_entity_cfg.joint_ids
        )

        self.reset_controllers()

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
        # compute ultrasound target pose
        self.get_US_target_pose()
        # compute joint positions
        # set joint positions
        self._move_towards_target(
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        self.get_drill_ee_pose_b()
        self.get_traj_to_tip_state()
        self.last_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)

        self.max_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        self.last_max_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        

        # if hasattr(self, "total_rewards") and torch.abs(self.total_rewards.mean()) > 0:
        #     wandb.log({"total_reward": self.total_rewards.mean().item()})
        #     wandb.log({"total_insertion": self.total_insertion.mean().item()})
        #     wandb.log({"last_sin": self.last_traj_to_tip_sin.mean().item()})
        #     wandb.log({"last_dist": self.last_tip_to_traj_dist.mean().item()})
        #     wandb.log({"last_angle": self.angle.mean().item()})
        #     wandb.log({"last_total_dist": self.total_dist.mean().item()})
        #     wandb.log({"last_insert_err": self.insert_err[self.inserted].mean().item()})

        self.last_tip_to_traj_dist = copy.deepcopy(self.tip_to_traj_dist)
        self.last_total_dist = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_traj_to_tip_sin = copy.deepcopy(self.traj_to_tip_sin)
        self.last_unsafe = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_safe_close = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_insertion = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_traj_pos_along_traj_safe_close = torch.ones(
            self.scene.num_envs, device=self.sim.device
        ) * (-self.safe_height)
        self.ever_unsafe = torch.zeros(self.scene.num_envs, device=self.sim.device)

        # if hasattr(self, "total_costs") and torch.abs(self.total_rewards.mean()) > 0:
        #     wandb.log({"total_cost": self.total_costs.mean().item()})
        self.total_rewards = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_costs = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_close = torch.zeros(self.scene.num_envs, device=self.sim.device)

        # record information
        ones = torch.ones((self.scene.num_envs,), device=self.sim.device)
        self.extras["traj_drct"] = self.vertebra_viewer.traj_drct
        self.extras["human_to_tip_pos"] = self.human_to_tip_pos
        self.extras["human_to_tip_quat"] = self.human_to_tip_quat
        self.extras["safe_height"] = self.safe_height * ones
        self.extras["traj_half_length"] = self.vertebra_viewer.traj_half_length
        self.extras["traj_radius"] = self.vertebra_viewer.traj_radius
        self.extras["tip_to_traj_dist"] = self.tip_to_traj_dist
        self.extras["traj_to_tip_sin"] = self.traj_to_tip_sin
        self.extras["human_to_traj_pos"] = self.vertebra_viewer.human_to_traj_pos
        self.extras["tip_pos_along_traj"] = self.tip_pos_along_traj
        self.extras["cost"] = torch.zeros(self.scene.num_envs, device=self.sim.device)

    def check_nan(self):
        if torch.isnan(self.US_ee_pos_b).any() or torch.isnan(self.US_ee_quat_b).any():
            print("US_ee_pos_b", self.US_ee_pos_b)
            print("US_ee_quat_b", self.US_ee_quat_b)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.drill_ee_pos_b).any()
            or torch.isnan(self.drill_ee_quat_b).any()
        ):
            print("drill_ee_pos_b", self.drill_ee_pos_b)
            print("drill_ee_quat_b", self.drill_ee_quat_b)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.human_to_tip_pos).any()
            or torch.isnan(self.human_to_tip_quat).any()
        ):
            print("human_to_tip_pos", self.human_to_tip_pos)
            print("human_to_tip_quat", self.human_to_tip_quat)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.tip_pos_along_traj).any()
            or torch.isnan(self.tip_to_traj_dist).any()
            or torch.isnan(self.traj_to_tip_sin).any()
        ):
            print("tip_pos_along_traj", self.tip_pos_along_traj)
            print("tip_to_traj_dist", self.tip_to_traj_dist)
            print("traj_to_tip_sin", self.traj_to_tip_sin)
            raise ValueError("nan value detected")
        if torch.isnan(self.total_rewards).any() or torch.isnan(self.total_costs).any():
            print("total_rewards", self.total_rewards)
            print("total_costs", self.total_costs)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.last_tip_pos_along_traj).any()
            or torch.isnan(self.last_tip_to_traj_dist).any()
            or torch.isnan(self.last_traj_to_tip_sin).any()
        ):
            print("last_tip_pos_along_traj", self.last_tip_pos_along_traj)
            print("last_tip_to_traj_dist", self.last_tip_to_traj_dist)
            print("last_traj_to_tip_sin", self.last_traj_to_tip_sin)
            raise ValueError("nan value detected")
        if torch.isnan(self.US_slicer.ct_img_tensor).any():
            print("ct_img_tensor", self.US_slicer.human_to_ee_target_pos)
            raise ValueError("nan value detected")
        if torch.isnan(self.US_slicer.label_img_tensor).any():
            print("label_img_tensor", self.US_slicer.human_to_ee_target_pos)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.US_to_drill_pos).any()
            or torch.isnan(self.US_to_drill_quat).any()
        ):
            print("US_to_drill_pos", self.US_to_drill_pos)
            print("US_to_drill_quat", self.US_to_drill_quat)
            raise ValueError("nan value detected")
