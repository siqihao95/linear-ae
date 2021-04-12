# Speeding up learning ordered principal componentswith linear autoencoders

This repository is the implementation of the CSC2541 (2021 winter) course project "Speeding up learning ordered principal componentswith linear autoencoders".

It is based on the official implementation of ["Regularized linear autoencoders recover the principal components, eventually"](https://github.com/XuchanBao/linear-ae).
## Requirements
Create a new conda environment:
```conda
conda create -n linear-autoencoders python=3.7
source activate linear-autoencoders
```

Install requirements:
```setup
pip install -r requirements.txt
```

## Training

We recommend using [Weights & Biases](https://www.wandb.com/) (`wandb`) for training and keeping track of results.

### Option 1: run a single experiment
To run an experiment with a single set of configurations, go to `mains/train_mnist.py` or `mains/train_synth.py`
and follow the comments to edit the configuration. Then run the following command to train on MNIST dataset:
```shell script
python -m mains.train_mnist
```
or on the synthetic dataset:
```shell script
python -m mains.train_synth
```

### Option 2: run batched experiments with wandb sweep
In `sweeps/sweep.sh`,
- Set `PYTHONPATH` to be the root path of this repository.
- Find your `wandb` API key in [settings](https://app.wandb.ai/settings), and paste it after `wandb login`.
- Enter your `wandb` username. You can find your username in the "profile" page under your name.

Then, create yaml files specifying the details of the sweep.

#### Example 1: MNIST experiment (wandb sweep)
First, create a [wandb sweep](https://docs.wandb.com/sweeps) with the following command
```sweep_mnist
wandb sweep sweeps/mnist.yaml
```
This should generate a sweep ID. Copy it and paste in `sweeps/sweep.sh`. You can now run the sweep agents.
```run_sweep_mnist
bash sweeps/sweep.sh
``` 
You can run multiple of the above command in parallel. Check your results [here](https://app.wandb.ai/).

#### Example 2: Synthetic dataset experiment
Similar to the MNIST experiment, first create a wandb sweep.
```sweep_synth
wandb sweep sweeps/synth.yaml
```
This should generate a sweep ID. Copy it and paste in `sweeps/sweep.sh`. You can now run the sweep agents.
```run_sweep_mnist
bash sweeps/sweep.sh
``` 
You can run multiple of the above command in parallel. Check your results [here](https://app.wandb.ai/).
