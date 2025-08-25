#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`
module load eth_proxy

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gres=gpumem:20g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END
#SBATCH --mail-user=None
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster_extension/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
