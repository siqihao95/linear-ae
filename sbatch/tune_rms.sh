#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v1,t4v2,p100
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-49%50
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/lae/linear-ae/sbatch'
list=(
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
"cd ${dir_path} ; bash tune_rms.sh"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
