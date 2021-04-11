import os
import torch
import torchvision
import numpy as np
import wandb
from models.data_generators import DataGeneratorPCA
from utils.train import train_models
from configs.utils import create_model_from_config, create_metric_config, update_config
from configs.mnist import optimal_lrs

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# os.environ["WANDB_MODE"] = "dryrun"

# if you don't wish to create a wandb sweep, you can directly edit the following parameters
# - model_name: one of ('uniform_sum', 'non_uniform_sum', 'rotation', 'nd', 'nd_exp', 'vae')
# - optimizer: one of ('SGD', 'Adam')
# default_hparams = dict(
#     hdim=20,
#     model_name="rotation",
#     # optimizer="RMSprop_subspace_only",
#     # optimizer="RMSprop_rotation_only",
#     # optimizer="RMSprop_full",
#     # rmsprop_alpha=0.99,
#     # rmsprop_momentum=0.99,
#     optimizer="SGD",
#     subspace_eps=1e-8,
#     rotation_eps=1e-8,
#     train_itr=30000,
#     seed=1234,
#     batch_size=-1,
#     tie_weights=True,
#     lr=0.003)

# default_hparams = dict(
#     hdim=20,
#     model_name="rotation",
#     # optimizer="RMSprop_subspace_only",
#     # optimizer="RMSprop_rotation_only",
#     optimizer="RMSprop_full",
#     rmsprop_alpha=0.9,
#     # rmsprop_momentum=0.99,
#     # optimizer="SGD",
#     subspace_eps=1e-8,
#     rotation_eps=1e-8,
#     train_itr=30000,
#     seed=1234,
#     batch_size=-1,
#     tie_weights=True,
#     lr=0.0002)

# default_hparams = dict(
#     hdim=20,
#     model_name="rotation",
#     # optimizer="RMSprop_subspace_only",
#     optimizer="RMSprop_rotation_only",
#     # optimizer="RMSprop_full",
#     rmsprop_alpha=0.01,
#     # rmsprop_momentum=0.99,
#     # optimizer="SGD",
#     subspace_eps=1e-8,
#     rotation_eps=1e-8,
#     train_itr=30000,
#     seed=1234,
#     batch_size=-1,
#     tie_weights=True,
#     lr=0.0002)

default_hparams = dict(
    hdim=20,
    model_name="rotation",
    optimizer="RMSprop_subspace_only",
    # optimizer="RMSprop_rotation_only",
    # optimizer="RMSprop_full",
    rmsprop_alpha=0.99,
    rmsprop_momentum=0.99,
    # optimizer="SGD",
    subspace_eps=1e-8,
    rotation_eps=1e-8,
    train_itr=30000,
    seed=1234,
    batch_size=-1,
    tie_weights=True,
    lr=0.002)

# default_hparams = dict(
#     hdim=20,
#     model_name="rotation",
#     optimizer="RMSprop_naive",
#     # optimizer="RMSprop_rotation_only",
#     # optimizer="RMSprop_full",
#     rmsprop_alpha=0.99,
#     rmsprop_momentum=0.9,
#     # optimizer="SGD",
#     subspace_eps=1e-8,
#     rotation_eps=1e-8,
#     train_itr=30000,
#     seed=1234,
#     batch_size=-1,
#     tie_weights=True,
#     lr=0.002)

wandb.init(project='lae-rms-0408', config=default_hparams)

config = update_config(optimal_lrs)
# config = wandb.config
# set random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# uncomment the following to save checkpoints
ckpt_dir = os.path.join('results', 'mnist', 'hdim{}'.format(config.hdim),
                        'lr{}'.format(config.lr), config.optimizer)
os.makedirs(ckpt_dir, exist_ok=True)

# Get MNIST data
input_dim = 28 * 28

mnist_data = torchvision.datasets.MNIST(
    root='../',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]))
# full batch
if config.batch_size is None or config.batch_size == -1:
    batch_size = len(mnist_data)
else:
    batch_size = config.batch_size

mnist_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=batch_size,
                                           shuffle=True)

_, (raw_data, __) = next(enumerate(mnist_loader))
raw_data = torch.squeeze(raw_data.view(-1, input_dim))

# Center the data, and find ground truth principle directions
data_mean = torch.mean(raw_data, dim=0)
centered_data = raw_data - data_mean

data = DataGeneratorPCA(input_dim, config.hdim, load_data=centered_data.numpy())
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=False)

# non-uniform L2 regularization parameters
reg_min = 0.1
reg_max = 0.9

init_scale = 0.0001
train_itr = config.train_itr

# create model config
model_config = create_model_from_config(config,
                                        input_dim,
                                        init_scale=init_scale,
                                        reg_min=reg_min,
                                        reg_max=reg_max)

# define metrics
metric_config, eval_metrics_list = create_metric_config(data, data_loader)

train_stats_hdim, _ = train_models(data_loader,
                                   train_itr,
                                   metric_config,
                                   model_configs=[model_config],
                                   eval_metrics_list=eval_metrics_list,
                                   ckpt_dir=ckpt_dir,
                                   tie_weights=config.tie_weights)

train_stats_hdim.close()
