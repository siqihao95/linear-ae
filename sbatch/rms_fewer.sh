#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2
#SBATCH --qos=high
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/lae/linear-ae'
list=(
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"
"cd ${dir_path} ; bash sweeps/sweep.sh"

)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
