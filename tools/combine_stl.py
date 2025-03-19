import os
from stl import mesh
import numpy as np

def combine_stl_files_in_subfolders(root_folder):
    """
    Combines all STL files within each subfolder of the root folder into a single STL file.
    
    Parameters:
        root_folder (str): Path to the root directory.
    """
    # Iterate over all subdirectories of the root folder
    for subfolder in next(os.walk(root_folder))[1]:
        subfolder_path = os.path.join(root_folder, subfolder)
        combined_mesh = None

        # Iterate over all files in the current subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.stl'):  # Check if the file is an STL file
                file_path = os.path.join(subfolder_path, file_name)
                print(f"Processing file: {file_path}")

                # Load the STL file
                current_mesh = mesh.Mesh.from_file(file_path)

                # Combine with the accumulated mesh
                if combined_mesh is None:
                    combined_mesh = current_mesh
                else:
                    combined_mesh = mesh.Mesh(
                        np.concatenate([combined_mesh.data, current_mesh.data])
                    )
        
        # Save the combined mesh if any STL files were processed
        if combined_mesh:
            output_path = os.path.join(subfolder_path, f"combined.stl")
            combined_mesh.save(output_path)
            print(f"Combined STL saved to: {output_path}")

# Path to the root folder containing subfolders with STL files
root_folder_path = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl"
combine_stl_files_in_subfolders(root_folder_path)