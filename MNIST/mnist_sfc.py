# %%
# simple fully-connected GAN
# 128 parameter for both discriminator and generator - 1 hidden layer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Hyperparameter
device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
lr = 3e-4  # call out for the best learning rate of adam
z_dim = 64  # dimension of the z input into the generator
image_dim = 784  # dimension of the mnist data -> flatten 28*28*1
batch_size = 32
num_epoch = 50

# Discriminator

# Generator
