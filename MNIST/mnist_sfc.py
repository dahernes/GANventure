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
batch_size = 32
num_epoch = 50
z_dim = 64  # dimension of the z input into the generator
image_dim = 784  # dimension of the mnist data -> flatten 28*28*1

# Discriminator
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
# https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html?highlight=leaky+relu#torch.nn.LeakyReLU
# the stability of the GAN game suffers if you have sparse gradients - avoid sparse gradients
# like ReLU, MaxPool - LeakyReLU have good results in both neural nets
# https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html?highlight=sigmoid#torch.nn.Sigmoid
class Discriminator(nn.Module):
    def __int__(self, input_features):
        super().__int__()
        self.disc = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LeakyReLU(0.2),  # variable
            nn.Linear(128, 1),  # 1 Output (range (0,1))
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# Generator
# https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html?highlight=tanh#torch.nn.Tanh
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, image_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# TODO method for calculating the average mean and std of the whole dataset (-> for Normalize)
# TODO check Normalize for output range ((-1,1) | tanh -- (0,1) | sigmoid) -> pref (-1,1) tanh for generator
