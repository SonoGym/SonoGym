#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="/home/yunkao/git/IsaacLabExtensionTemplate/scripts/convert_stl_headless.py"

# Base directory for input files
INPUT_ROOT="/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_stl"

# Output directory
OUTPUT_ROOT="/home/yunkao/git/IsaacLabExtensionTemplate/exts/spinal_surgery/spinal_surgery/assets/data/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Ensure the output root directory exists
mkdir -p "$OUTPUT_ROOT"

# Process each file recursively
find "$INPUT_ROOT" -type f | while read -r input_file; do

    relative_path="${input_file#$INPUT_ROOT/}"

    output_base="${OUTPUT_ROOT}/${relative_path%.*}"
    output_subdir="${output_base}"
    output_file="${output_subdir}/$(basename ${relative_path%.*}).usd"
    
    # Ensure the output subdirectory exists
    mkdir -p "$output_subdir"
    
    # Run the Python script
    echo "Processing $input_file -> $output_file"
    python3 "$PYTHON_SCRIPT" "$input_file" "$output_file" --make-instanceable --mass 0.1 --collision-approximation "none"
    
    # Check if the Python script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $input_file"
    else
        echo "Successfully processed $input_file"
    fi
done

echo "All files processed."