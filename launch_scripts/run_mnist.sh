#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p t4v2
#SBATCH --qos=high
#SBATCH -c 8
#SBATCH --mem=16GB
#SBATCH --job-name=linear-ae-mnist
#SBATCH --output=out/mnist.out

export PYTHONPATH=/h/jennybao/projects/NNTD/linear-ae
python -m mains.train_mnist