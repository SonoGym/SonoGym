import nibabel as nib
import pyvista as pv
import numpy as np

# Load the NIfTI file
nii_file = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0016/combined_label_map.nii.gz"  # Replace with your .nii.gz file path
stl_mesh = "/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0016/body_trunc.stl"
nii_data = nib.load(nii_file)

# Extract the image data as a NumPy array
image_data = nii_data.get_fdata()
body_mesh = pv.read(stl_mesh)

# Visualize the grid
plotter = pv.Plotter()
plotter.show_axes_all()
image_data[image_data == 120] = 0.5
plotter.add_volume(
    image_data, 
)
plotter.add_mesh(body_mesh, color="white", opacity=0.5)
print(np.unique(image_data))
plotter.show()