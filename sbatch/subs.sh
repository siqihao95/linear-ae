#!/bin/bash
#SBATCH --partition=p100,t4v1
#SBATCH --gres=gpu:1
#SBATCH --qos=legacy
#SBATCH --account=legacy
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-19%20
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/lae/linear-ae'
list=(
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
"cd ${dir_path} ; bash sweeps/subs.sh"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
