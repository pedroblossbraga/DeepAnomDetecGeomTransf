#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:2             # Request n A100 GPUs
#SBATCH --partition=a100             # GPU partition
#SBATCH --time=15:00:00               # Max runtime 
#SBATCH --output=logs/job_output_%j.log   # Capture stdout to a log file
#SBATCH --error=logs/job_error_%j.log     # Capture stderr to a log file
#SBATCH --export=NONE                # Unset exported variables

unset SLURM_EXPORT_ENV

module load python

# initialize conda
eval "$(/home/hpc/iwai/iwai120h/miniconda3/bin/conda shell.bash hook)"

conda activate poetry_env
# conda activate tensorflow

# Add globally installed Poetry to PATH
export PATH="/home/hpc/iwai/iwai120h/.local/bin:$PATH"

# Record start time
START_TIME=$(date +%s)

# run script
# poetry run python gpu_version.py
poetry run python main.py

# Record end time and compute total duration
END_TIME=$(date +%s)
echo "Total Execution Time: $(($END_TIME - $START_TIME)) seconds"