# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.kinematics.gt_motion_generator import GTMotionGenerator, GTDiscreteMotionGenerator
from spinal_surgery.lab.agents.imitation_agents_discrete import ImitationAgent, PositionAgent
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR
import wandb
import numpy as np


scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/robotic_US_imitation.yaml", 'r'))

# robot
robot_cfg = scene_cfg['robot']
INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_cfg['joint_pos'][0],
        "lbr_joint_1": robot_cfg['joint_pos'][1],
        "lbr_joint_2": robot_cfg['joint_pos'][2],
        "lbr_joint_3": robot_cfg['joint_pos'][3], # -1.2,
        "lbr_joint_4": robot_cfg['joint_pos'][4],
        "lbr_joint_5": robot_cfg['joint_pos'][5], # 1.5,
        "lbr_joint_6": robot_cfg['joint_pos'][6],
    },
    pos = robot_cfg['pos'] # ((0.0, -0.75, 0.4))
)

# patient
patient_cfg = scene_cfg['patient']
quat = R.from_euler("yxz", patient_cfg['euler_yxz'], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg['pos'], # 0.7
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

# bed
bed_cfg = scene_cfg['bed']
quat = R.from_euler("xyz", bed_cfg['euler_xyz'], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg['pos'], 
    rot=((quat[3], quat[0], quat[1], quat[2]))
)
scale_bed = bed_cfg['scale']
# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_body_from_urdf/" + p_id for p_id in patient_cfg['id_list']
]
human_stl_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/" + p_id for p_id in patient_cfg['id_list']
]
human_raw_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset/" + p_id for p_id in patient_cfg['id_list']
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg['label_res']
scale = 1/label_res

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # kuka US
    robot_US = KUKA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_US",
        init_state=INIT_STATE_ROBOT_US
    )

    # medical bad
    medical_bad = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_no_contact/hospital_bed.usd",
            scale = (scale_bed, scale_bed, scale_bed),
        ),
        init_state = INIT_STATE_BED
    )


    human = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Human", 
        spawn=sim_utils.MultiUsdFileCfg(
        usd_path=usd_file_list,
        random_choice=False,
        scale = (label_res, label_res, label_res),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           articulation_enabled=False,
        ),
        ),
        init_state = INIT_STATE_HUMAN,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot_US"]
    human = scene['human']
    robot_entity_cfg = SceneEntityCfg("robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # define ik controllers
    ik_params = {"lambda_val": 0.00001}
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, scene.num_envs, device=sim.device)
    pose_diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    pose_diff_ik_controller = DifferentialIKController(pose_diff_ik_cfg, scene.num_envs, device=sim.device)

    # construct label image slicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", 'r'))
    
    sim_cfg = scene_cfg['sim']
    motion_plan_cfg = scene_cfg['motion_planning']
    init_cmd_pose = torch.tensor(sim_cfg['patient_xz_init'], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)
    goal_cmd_pose = torch.tensor(motion_plan_cfg['patient_xz_goal'], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)

    # construct US simulator
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", 'r'))
    US_slicer = USSlicer(
        us_cfg,
        label_map_list, 
        ct_map_list,
        sim_cfg['if_use_ct'],
        human_stl_list,
        scene.num_envs, 
        sim_cfg['patient_xz_range'], 
        sim_cfg['patient_xz_init'], 
        sim.device, 
        label_convert_map,
        us_cfg['image_size'], 
        us_cfg['resolution'],
        visualize=sim_cfg['vis_seg_map'],
    )
    gt_motion_generator = GTDiscreteMotionGenerator(
        goal_cmd_pose=goal_cmd_pose,
        scale=torch.tensor(motion_plan_cfg['scale'], device=sim.device),
        num_envs=scene.num_envs,
        surface_map_list=US_slicer.surface_map_list,
        surface_normal_list=US_slicer.surface_normal_list,
        label_res=label_res,
        US_height=US_slicer.height,
    )

    # learning
    cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/agents/cfgs/imitation_agent_cfg.yaml", 'r'))
    agent = ImitationAgent(cfg, scene.num_envs, sim.device, US_slicer.img_size)

    train_cfg = scene_cfg['train']
    val_cfg = scene_cfg['validation']

    wandb.init(project='imitation_navigation', config=cfg)
    

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    episode = 0
    # Simulation loop
    while simulation_app.is_running() and episode <= train_cfg['num_episodes']:
        # Reset
        if count % sim_cfg['episode_length'] == 0:
            episode += 1
            # reset counter
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            diff_ik_controller.reset()
            pose_diff_ik_controller.reset()

            # get ee pose in base frame
            US_root_pose_w = robot.data.root_state_w[:, 0:7]

            US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
            # compute frame in root frame
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                US_root_pose_w[:, 0:3], US_root_pose_w[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
            )

            ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
            diff_ik_controller.set_command(ik_commands, US_ee_pos_b, US_ee_quat_b)
            ik_commands_pose = torch.zeros(scene.num_envs, pose_diff_ik_controller.action_dim, device=sim.device)
            pose_diff_ik_controller.set_command(ik_commands_pose, US_ee_pos_b, US_ee_quat_b)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

            US_slicer.update_cmd(init_cmd_pose - US_slicer.current_x_z_x_angle_cmd)

            # train
            if count>0 and not episode % val_cfg['val_interval']==1:

                agent.train()

            agent.reset()
            hidden = torch.zeros((scene.num_envs, 2, agent.hidden_size), device=sim.device)
            last_vector = torch.zeros((scene.num_envs, 1, 3), device=sim.device)



        start = time.time()

        # get human frame
        human_world_poses = human.data.root_state_w # these are already the initial poses
        # define world to human poses
        world_to_human_pos, world_to_human_rot = human_world_poses[:, 0:3], human_world_poses[:, 3:7]

        # update the view
        # get ee pose in wolrd frame
        US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
        US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
        )
        # update image simulation
        US_slicer.slice_US(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        if sim_cfg['vis_us']:
            US_slicer.visualize()

        # go to the init pose
        if count % sim_cfg['episode_length'] < train_cfg['rand_steps']:
            # Apply random action # cover tha init state
            if not episode % val_cfg['val_interval']==0:
                rand_scale = torch.tensor(train_cfg['rand_init_scale'], device=sim.device).reshape((1, -1))
                rand_start = torch.tensor(train_cfg['rand_init_start'], device=sim.device).reshape((1, -1))
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device)*rand_scale + rand_start
            else:
                rand_scale = torch.tensor(val_cfg['rand_init_scale'], device=sim.device).reshape((1, -1))
                rand_start = torch.tensor(val_cfg['rand_init_start'], device=sim.device).reshape((1, -1))
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device)*rand_scale + rand_start

            # rand_x_z_angle = torch.zeros((scene.num_envs, 3), device=sim.device)
            US_slicer.update_cmd(rand_x_z_angle-US_slicer.current_x_z_x_angle_cmd)
        else:
            # get cur command
            us_imgs = US_slicer.us_img_tensor.unsqueeze(1).unsqueeze(1).float() / 255.0 # (num_envs, 1, 1, w, h)

            output, next_hidden = agent.predict(images=us_imgs, vectors=last_vector, hidden_states=hidden.transpose(0, 1).contiguous())
            direction = (output >= 0).int() * 2 - 1
            cur_cmd = direction * gt_motion_generator.scale

            # convert current pose to human command pose
            cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
            )
            cur_cmd_pose = gt_motion_generator.human_cmd_state_from_ee_pose(cur_human_ee_pos, cur_human_ee_quat)
            
            gt_cmd, gt_cmd_pose = gt_motion_generator.generate_gt_human_cmd(cur_cmd_pose)
            gt_output = (gt_cmd >= 0).int()
            # print('GT Command:', gt_cmd)
            print('cur_cmd_pose', cur_cmd_pose[:10,:])
            print('episode', episode)
            # print('gt_cmd', gt_cmd[:5,:])

            # update buffer
            agent.record(images=us_imgs, vectors=last_vector, hidden_states=hidden, gt_output=gt_output)

            diff_cmd = cur_cmd + cur_cmd_pose-US_slicer.current_x_z_x_angle_cmd

            last_vector = output.detach().unsqueeze(1)
            hidden = next_hidden.detach().transpose(0, 1)
            
            if episode % val_cfg['val_interval']==0:
                US_slicer.update_cmd(diff_cmd)
                if count % sim_cfg['episode_length'] == sim_cfg['episode_length']-1:
                    mean_error = torch.mean(torch.abs(cur_cmd_pose - goal_cmd_pose))
                    wandb.log({'episode':episode, 'mean_error': mean_error})
            # Apply random action
            else: 
                # random
                # gt
                diff_cmd = gt_cmd_pose-US_slicer.current_x_z_x_angle_cmd
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * train_cfg['motion_noise_scale'] - train_cfg['motion_noise_scale'] / 2
                rand_x_z_angle[:, 2] = (rand_x_z_angle[:, 2] / 10)
                diff_cmd = diff_cmd + rand_x_z_angle
                US_slicer.update_cmd(diff_cmd)
            
        
        # compute frame in root frame
        world_to_ee_target_pos, world_to_ee_target_rot = US_slicer.compute_world_ee_pose_from_cmd(
            world_to_human_pos, world_to_human_rot)
        world_to_ee_target_pose = torch.cat([world_to_ee_target_pos, world_to_ee_target_rot], dim=-1)
        
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], world_to_ee_target_pos, world_to_ee_target_rot
        )
        base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

        # set new command
        pose_diff_ik_controller.set_command(base_to_ee_target_pose)
        
        # get joint position targets
        US_jacobian = robot.root_physx_view.get_jacobians()[:, US_ee_jacobi_idx-1, :, robot_entity_cfg.joint_ids]
        US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = pose_diff_ik_controller.compute(
            US_ee_pos_b, 
            US_ee_quat_b,
            US_jacobian, 
            US_joint_pos
        )
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        end = time.time()
        # print(f"Time taken for step: {end - start}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device) # , gravity=[0.0, 0.0, 0.0]
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    robot_scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(robot_scene_cfg)
    # load label maps
    label_map_list = []
    for label_map_file in label_map_file_list:
        label_map = nib.load(label_map_file).get_fdata()
        label_map_list.append(label_map)
    # load ct maps
    ct_map_list = []
    for ct_map_file in ct_map_file_list:
        ct_map = nib.load(ct_map_file).get_fdata()
        ct_min_max = scene_cfg['sim']['ct_range']
        ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
        ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
        ct_map_list.append(ct_map)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    # run the main function
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    # close sim app
    simulation_app.close()