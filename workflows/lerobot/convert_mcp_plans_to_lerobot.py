"""Convert multi-contact plan trajectories to LeRobot format.

This script loads multi-contact plan trajectories and play them kinematically in the Isaac Sim
simulator. The script then converts the trajectories to LeRobot format.

.. code-block:: bash

    python scripts/lerobot/convert_mcp_plans_to_lerobot.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# parse the arguments
argparser = argparse.ArgumentParser(description="Convert multi-contact plans to LeRobot format.")
argparser.add_argument("--spacing", type=float, default=2.5, help="Spacing between the robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(argparser)
# parse the arguments
args_cli = argparser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: F401

from isaac.locoma import LOCOMA_DATA_DIR  # noqa: F401
from isaac.locoma.assets import DoorObject, DoorObjectCfg
from isaac.locoma.datasets import MultiContactPlanDataLoader, MultiContactPlanDataLoaderCfg

##
# Pre-defined configs
##
from isaac.locoma.assets.config import DOOR_TYPE_CONFIG_DICT  # isort:skip
from isaaclab_assets.robots.alma import ALMA_D_HOOK_TOOL_CFG  # isort:skip


@configclass
class LocomaSceneCfg(InteractiveSceneCfg):
    """Configuration for the door scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = ALMA_D_HOOK_TOOL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", collision_group=1)

    # doors
    door: DoorObjectCfg = DOOR_TYPE_CONFIG_DICT["nominal_horizontal"].replace(
        prim_path="{ENV_REGEX_NS}/Door", collision_group=2
    )

    # sensors
    rs_cam_front_upper = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/rs_cam_front_upper",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.47, 0.0, 0.0),
            rot=(0.99144, 0.0, 0.13052, 0.0),
            convention="world",
        ),
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        width=424,
        height=240,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[227.34, 0.0, 212.5, 0.0, 227.34, 120.5, 0.0, 0.0, 1.0],
            width=424,
            height=240,
            clipping_range=(0.01, 2.5),
        ),
    )


MCP_TASK = "Open a dishwasher door completely while switching contact points."
MCP_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (25,),
        "names": {
            "axes": [
                "root_pos_init_object_x",
                "root_pos_init_object_y",
                "root_pos_init_object_z",
                "root_quat_init_object_w",
                "root_quat_init_object_x",
                "root_quat_init_object_y",
                "root_quat_init_object_z",
                "joint_pos_LF_HAA",
                "joint_pos_LH_HAA",
                "joint_pos_RF_HAA",
                "joint_pos_RH_HAA",
                "joint_pos_SH_ROT",
                "joint_pos_LF_HFE",
                "joint_pos_LH_HFE",
                "joint_pos_RF_HFE",
                "joint_pos_RH_HFE",
                "joint_pos_SH_FLE",
                "joint_pos_LF_KFE",
                "joint_pos_LH_KFE",
                "joint_pos_RF_KFE",
                "joint_pos_RH_KFE",
                "joint_pos_EL_FLE",
                "joint_pos_FA_ROT",
                "joint_pos_WRIST_1",
                "joint_pos_WRIST_2",
            ],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (25 + 1 + 1,),
        "names": {
            "axes": [
                "root_pos_init_object_x",
                "root_pos_init_object_y",
                "root_pos_init_object_z",
                "root_quat_init_object_w",
                "root_quat_init_object_x",
                "root_quat_init_object_y",
                "root_quat_init_object_z",
                "joint_pos_LF_HAA",
                "joint_pos_LH_HAA",
                "joint_pos_RF_HAA",
                "joint_pos_RH_HAA",
                "joint_pos_SH_ROT",
                "joint_pos_LF_HFE",
                "joint_pos_LH_HFE",
                "joint_pos_RF_HFE",
                "joint_pos_RH_HFE",
                "joint_pos_SH_FLE",
                "joint_pos_LF_KFE",
                "joint_pos_LH_KFE",
                "joint_pos_RF_KFE",
                "joint_pos_RH_KFE",
                "joint_pos_EL_FLE",
                "joint_pos_FA_ROT",
                "joint_pos_WRIST_1",
                "joint_pos_WRIST_2",
                "joint_pos_door_hinge",
                "phase",
            ],
        },
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (1,),
        "names": [
            "joint_pos_door_hinge",
        ],
    },
    "observation.images.ultrasound": {
        "dtype": "video",
        "shape": (3, 240, 424),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
    "observation.images.ct_scan": {
        "dtype": "video",
        "shape": (3, 240, 424),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
}


@torch.no_grad()
def main():
    # Load the data loader configuration
    data_cfg = MultiContactPlanDataLoaderCfg(
        # data_dir=f"{LOCOMA_DATA_DIR}/mcp_trajectories",
        # data_dir="/home/mayank/Datasets/guided-mcp/plans-txt-cleaned",
        # data_dir="/workspace/rsl_assets/plans-txt",
        data_dir="/media/vulcan/Datasets/guided-mcp/plans-txt-cleaned",
        task_name=[
            "horizontal_door_opening",
            # "horizontal_door_closing",
            # "right_hinge_door_pulling",
            # "left_hinge_door_pushing",
        ],
        speed_factor=1.0,
        skip_frames=0,
    )
    # Create the data loader
    data_loader = MultiContactPlanDataLoader(data_cfg, device=args_cli.device)

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=args_cli.device))
    # Set main camera
    sim.set_camera_view(eye=[-3.6, 4.6, 2.5], target=[1.0, 0.0, 0.5])

    # Design the scene
    # TODO: Make this better with parallelized scenes. The recorder isn't so flexible yet.
    scene = InteractiveScene(cfg=LocomaSceneCfg(num_envs=1, env_spacing=args_cli.spacing, replicate_physics=False))

    # Play the simulator
    sim.reset()

    # extract robot
    robot: Articulation = scene["robot"]
    door: DoorObject = scene["door"]
    rs_cam_front_upper: TiledCamera = scene["rs_cam_front_upper"]

    # rearrange the joint indices in the trajectory to match the simulation ordering
    data_loader.rearrange_joint_trajectory_indices(robot.data.joint_names)

    # Create the data collector
    lerobot_dataset = LeRobotDataset.create(
        repo_id="rsl/horizontal-door-opening",
        fps=int(1 / data_loader.traj_dt),
        root="/media/vulcan/Datasets/guided-mcp/lerobot-dataset",
        robot_type="alma-d-hook",
        features=MCP_FEATURES,
    )

    # Find end-effector indices
    ee_indices, ee_names = robot.find_bodies(name_keys=[".*_FOOT", ".*_END_EFFECTOR"])

    print("Robot end-effectors:")
    for ee_name, ee_index in zip(ee_names, ee_indices):
        print(f"  - {ee_name} (index: {ee_index})")
    print("Robot joints:")
    for joint_index, joint_name in enumerate(robot.data.joint_names):
        print(f"  - {joint_name} (index: {joint_index})")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    print("Saving dataset to HDF5...")
    print(f"  - {lerobot_dataset.meta.root}")

    # Simulate physics
    for ep_index in tqdm.tqdm(range(data_loader.num_traj)):
        # get the number of steps in the episode
        num_steps = data_loader.traj_lengths[ep_index].item()

        for step_index in range(num_steps - 1):
            # set the scene state from the data loader
            # -- robots
            root_state = data_loader.robot_root_state_w[ep_index, step_index].clone().unsqueeze(0)
            root_state[..., :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # joint state
            joint_pos = data_loader.robot_joint_pos[ep_index, step_index].clone().unsqueeze(0)
            joint_vel = data_loader.robot_joint_vel[ep_index, step_index].clone().unsqueeze(0)
            joint_vel[:] = 0.0
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # joint target
            robot.set_joint_position_target(joint_pos)
            # -- door
            door_joint_pos = data_loader.object_joint_pos[ep_index, step_index].clone().unsqueeze(0)
            door_joint_vel = data_loader.object_joint_vel[ep_index, step_index].clone().unsqueeze(0)
            door_joint_vel[:] = 0.0
            door.write_joint_state_to_sim(door_joint_pos, door_joint_vel)

            # apply buffers to sim
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            scene.update(sim_dt)

            # read out values from the robot and check everything is good
            # this is to make sure that the stepping did not change anything
            # robot_root_pose = robot.data.root_state_w[:, :7]
            # robot_root_pose[:, :3] -= scene.env_origins
            # robot_joint_pos = robot.data.joint_pos
            # object_joint_pos = door.data.joint_pos[:, :1]
            # torch.testing.assert_close(robot_root_pose, data_loader.robot_root_state_w[:, count, :7])
            # torch.testing.assert_close(robot_joint_pos, data_loader.robot_joint_pos[:, count])
            # torch.testing.assert_close(object_joint_pos, data_loader.object_joint_pos[:, count])

            obs_dict = {
                "root_pos_init_object": data_loader.robot_root_pos_init_object[ep_index, step_index].unsqueeze(0),
                "root_quat_init_object": data_loader.robot_root_quat_init_object[ep_index, step_index].unsqueeze(0),
                "robot_joint_pos": data_loader.robot_joint_pos[ep_index, step_index].unsqueeze(0),
                "object_joint_pos": data_loader.object_joint_pos[ep_index, step_index].unsqueeze(0),
            }
            for ee_name, ee_index in zip(ee_names, ee_indices):
                # obtain the end-effector pose
                ee_pose_w = robot.data.body_state_w[:, ee_index, :7]
                root_pose_w = robot.data.root_state_w[:, :7]
                # compute the relative pose
                ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                    root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
                )
                obs_dict[f"{ee_name}_pos_base"] = ee_pos_b
                obs_dict[f"{ee_name}_quat_base"] = ee_quat_b

            # next step
            root_pos_init_object_next = data_loader.robot_root_pos_init_object[ep_index, step_index + 1].unsqueeze(0)
            root_quat_init_object_next = data_loader.robot_root_quat_init_object[ep_index, step_index + 1].unsqueeze(0)

            # obtain the relative pose change
            root_pos_init_object_rel, root_quat_init_object_rel = math_utils.subtract_frame_transforms(
                obs_dict["root_pos_init_object"],
                obs_dict["root_quat_init_object"],
                root_pos_init_object_next,
                root_quat_init_object_next,
            )
            robot_joint_pos_next = data_loader.robot_joint_pos[ep_index, step_index + 1].unsqueeze(0)
            object_joint_pos_next = data_loader.object_joint_pos[ep_index, step_index + 1].unsqueeze(0)

            # create a dictionary of actions
            act_dict = {
                # absolute values
                "root_pos_init_object_abs": root_pos_init_object_next,
                "root_quat_init_object_abs": root_quat_init_object_next,
                "robot_joint_pos_abs": robot_joint_pos_next,
                "object_joint_pos_abs": object_joint_pos_next,
                # relative values
                "root_pos_init_object_rel": root_pos_init_object_rel,
                "root_quat_init_object_rel": root_quat_init_object_rel,
                "robot_joint_pos_rel": robot_joint_pos_next - obs_dict["robot_joint_pos"],
                "object_joint_pos_rel": object_joint_pos_next - obs_dict["object_joint_pos"],
                # general metadata
                "phase": (step_index / data_loader.traj_lengths[ep_index]).clip(0.0, 1.0).view(-1, 1),
            }

            # create a tensor of observations.state
            obs_state = torch.cat(
                [
                    obs_dict["root_pos_init_object"],
                    obs_dict["root_quat_init_object"],
                    obs_dict["robot_joint_pos"],
                ],
                dim=-1,
            )
            obs_env_state = torch.cat(
                [
                    obs_dict["object_joint_pos"],
                ],
                dim=-1,
            )

            # create a tensor of actions
            actions = torch.cat(
                [
                    act_dict["root_pos_init_object_abs"],
                    act_dict["root_quat_init_object_abs"],
                    act_dict["robot_joint_pos_abs"],
                    act_dict["object_joint_pos_abs"],
                    act_dict["phase"],
                ],
                dim=-1,
            )

            # read out the images
            front_cam_depth = rs_cam_front_upper.data.output["distance_to_image_plane"].clone()
            front_cam_semantic = rs_cam_front_upper.data.output["semantic_segmentation"].clone()

            # squeeze the images
            front_cam_depth = front_cam_depth.squeeze(0)
            front_cam_semantic = front_cam_semantic.squeeze(0)[:, :, :3]
            # convert the images to uint8
            z_near, z_far = 0.1, 1.5
            front_cam_depth = front_cam_depth.clip(z_near, z_far)
            front_cam_depth = (front_cam_depth / (z_far - z_near) * 255.0).to(torch.uint8).repeat(1, 1, 3)
            front_cam_semantic = front_cam_semantic.to(torch.uint8)

            # create a dictionary of frame data
            frame = {
                "action": actions.squeeze(0).cpu(),
                "observation.state": obs_state.squeeze(0).cpu(),
                "observation.environment_state": obs_env_state.squeeze(0).cpu(),
                "observation.images.front_upper_depth": front_cam_depth.cpu(),
                "observation.images.front_upper_semantic": front_cam_semantic.cpu(),
                "task": MCP_TASK,
            }
            lerobot_dataset.add_frame(frame)

        # save the episode
        lerobot_dataset.save_episode()


if __name__ == "__main__":
    main()
    simulation_app.close()
