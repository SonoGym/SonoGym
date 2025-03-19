#!/bin/bash

ROOT_DIR="/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl"

# Iterate through each subdirectory of the root directory
for dir in "$ROOT_DIR"/*/; do
    # Define the input and output file paths
    input_file="$dir/combined.stl"
    output_file="$dir/combined_wrapwrap.stl"
    
    # Check if 'combined.stl' exists
    if [ -f "$input_file" ]; then
        echo "Processing: $input_file"
        # Run wrapwrap command
        /home/yunkao/git/wrapwrap/build/wrapwrap -i "$input_file" -alpha 300 -offset 600 -o "$output_file"
        echo "Output saved to: $output_file"
    else
        echo "Skipping: No combined.stl found in $dir"
    fi
done