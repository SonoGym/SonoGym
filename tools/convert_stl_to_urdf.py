import os

def generate_urdf(stl_path, urdf_path, link_name="base_link", color=(0.7, 0.7, 0.7, 1.0)):
    """
    Generates a URDF file referencing the given STL file with a specified color.

    Parameters:
        stl_path (str): Path to the STL file.
        urdf_path (str): Path to save the generated URDF file.
        link_name (str): Name of the base link (default: "base_link").
        color (tuple): RGBA color for the visual mesh (default: gray (0.7, 0.7, 0.7, 1.0)).
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file '{stl_path}' not found!")

    # Extract filename
    stl_filename = os.path.basename(stl_path)

    # Ensure color values are within range [0, 1]
    r, g, b, a = [max(0, min(1, c)) for c in color]

    urdf_content = f"""<?xml version="1.0"?>
<robot name="{link_name}_robot">
    <link name="{link_name}">
        <visual>
            <geometry>
                <mesh filename="{stl_filename}" scale="1 1 1"/>
            </geometry>
            <material name="custom_color">
                <color rgba="{r} {g} {b} {a}"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <mesh filename="{stl_filename}" scale="1 1 1"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="60.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
    </link>
</robot>
"""

    with open(urdf_path, "w") as urdf_file:
        urdf_file.write(urdf_content)

    print(f"✅ URDF file saved: {urdf_path}")

def process_patients(root_dir, output_root, color=(0.7, 0.7, 0.7, 1.0)):
    """
    Iterates over subdirectories (patients) in the root directory, finds 'combine_wrapwrap.stl',
    and converts it to a URDF in the output root directory.

    Parameters:
        root_dir (str): Root directory containing patient subdirectories.
        output_root (str): Root directory where URDF files will be saved.
        color (tuple): RGBA color for the mesh in the URDF.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for patient in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_dir):
            continue  # Skip if not a directory

        stl_path = os.path.join(patient_dir, "combined_wrapwrap.stl")
        if os.path.exists(stl_path):
            # Ensure output directory exists
            output_patient_dir = os.path.join(output_root, patient)
            os.makedirs(output_patient_dir, exist_ok=True)

            # Define URDF output path
            urdf_path = os.path.join(output_patient_dir, "combined_wrapwrap.urdf")

            # Generate the URDF
            generate_urdf(stl_path, urdf_path, color=color)
        else:
            print(f"❌ No STL file found for patient: {patient}")

# Example usage
input_root = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl"  # Replace with actual path
output_root = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl"  # Replace with actual path

custom_color = (0.7, 0.5, 0.2, 0.7)  # Example: Red color with full opacity

process_patients(input_root, output_root, color=custom_color)