import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
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
import os

from pxr import Usd, UsdGeom, Sdf, UsdPhysics, Gf
import carb
import shutil

def create_robot_usd_with_physics(output_path, link_usd_files):
    """
    Combine multiple link USD files into a single robot USD file with USDPhysics joints.

    Args:
        output_path (str): Path to save the output robot USD file.
        link_usd_files (list): List of paths to USD files representing the robot's links.
        joints_config (list): List of dictionaries defining joints. Each dictionary includes:
            - "parent": (str) Path to the parent prim.
            - "child": (str) Path to the child prim.
            - "type": (str) Type of joint, e.g., "revolute", "fixed", "prismatic".
            - "axis": (list) Optional, axis of movement/rotation [x, y, z].
            - "position": (list) Optional, [x, y, z] position of the joint in world coordinates.
    """
    # Create a new USD stage
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    # Create a root Xform node
    base_prim = stage.DefinePrim(f"/Robot", "Xform")
    stage.SetDefaultPrim(base_prim)
    root_path = base_prim.GetPath()
    last_link_name = "Robot"

    # Load and reference each link USD file
    for idx, link_file in enumerate(link_usd_files):
        link_name = os.path.splitext(os.path.basename(link_file))[0]
        link_path = f"{root_path}/{link_name}"
        link_prim = stage.DefinePrim(link_path, "Xform")
        link_prim.GetReferences().AddReference(link_file)
        # UsdPhysics.CollisionAPI.Apply(link_prim).GetCollisionEnabledAttr().Set(False)
        

        # add fix joint
        if idx==0:
            # # Add joints between links
            # joint_name = f"Joint_{last_link_name}_{link_name}"
            # joint_path = f"{root_path}/{joint_name}"
            # # Define the joint prim
            # joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)
            # # Set parent and child links
            # parent_path = f"{root_path}"
            # child_path = f"{root_path}/{link_name}"
            pass

        else:
            
            # Add joints between links
            joint_name = f"Joint_{last_link_name}_{link_name}"
            joint_path = f"{root_path}/{link_name}/{joint_name}"
            # Define the joint prim
            joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)
            # Set parent and child links
            parent_path = f"{root_path}/{last_link_name}"
            child_path = f"{root_path}/{link_name}"

            joint_prim.CreateBody0Rel().SetTargets([parent_path])
            joint_prim.CreateBody1Rel().SetTargets([child_path])
            # Set the joint position
            position = [0, 0, 0]  # Default position is at origin
            UsdGeom.Xform(joint_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*position))
            
        last_link_name = link_name
            
        
    # Save the USD stage
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    stage.Export(output_path)

    print(f"Robot USD file with physics joints saved to: {output_path}")

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def compute_combined_usd_for_subdirectories(root_path):
    """
    Combine all link USD files in each subdirectory of the given root path into a single robot USD file.

    Args:
        root_path (str): Path to the root directory containing subdirectories with link USD files.
    """
    # Get all subdirectories
    subdirectories = next(os.walk(root_path))[1]
    for subdirectory in subdirectories:
        subdirectory = os.path.join(root_path, subdirectory)
        print(f"Processing subdirectory: {subdirectory}")
        subfiles = get_all_file_paths(subdirectory)
        # Get all link USD files in the subdirectory
        link_usd_files = [f for f in subfiles if 'combined' not in f and f.endswith(".usd") and 'instanceable' not in f]
        # Create the robot USD file with physics joints
        output_usd_file = os.path.join(subdirectory, "combined", "combined.usd")
        create_robot_usd_with_physics(output_usd_file, link_usd_files)
        
        # link_folders = [f for f in subfiles if 'combined' in f]
        # for link_folder in link_folders:
        #     try:
        #         shutil.rmtree(link_folder)
        #     except OSError as e:
        #         print("Error: %s - %s." % (e.filename, e.strerror))


# Example Usage
if __name__ == "__main__":
    input_usd_files = [
        "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0021/brain/brain.usd",
        "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0021/vertebrae_C7/vertebrae_C7.usd",
        '/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0000/body_trunc/body_trunc.usd',
    ]
    output_usd_file = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd/s0021/combined/combined.usd"
    
    # Joint configuration
    root_path = '/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd_mix'

    create_robot_usd_with_physics(output_usd_file, input_usd_files)

    compute_combined_usd_for_subdirectories(root_path)