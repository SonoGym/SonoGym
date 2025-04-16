#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`
module load eth_proxy

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###

k=10  # Change this to the number of jobs you want
for ((i=0; i<k; i++)); do
    sbatch << EOF
#!/bin/bash

#SBATCH --gres=gpumem:20g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=23:50:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END
#SBATCH --mail-user=None
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
# modify the CUDA_VISIBLE_DEVICES variable to use the GPU you want
# export CUDA_VISIBLE_DEVICES=$i

bash "$1/docker/cluster_extension/run_singularity_sweep.sh" "$1" "$2" "${@:3}"
EOF
done


