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

# Discriminator
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
# https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html?highlight=leaky+relu#torch.nn.LeakyReLU
# the stability of the GAN game suffers if you have sparse gradients - avoid sparse gradients
# like ReLU, MaxPool - LeakyReLU have good results in both neural nets
# https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html?highlight=sigmoid#torch.nn.Sigmoid


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
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


# Hyperparameter
device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
lr = 3e-4  # call out for the best learning rate of adam
batch_size = 32
num_epoch = 50
z_dim = 64  # dimension of the z input into the generator
image_dim = 784  # dimension of the mnist data -> flatten 28*28*1 -> 784

# load the classes into the device
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# create noise
fix_noise = torch.randn((batch_size, z_dim)).to(device)

# dataset - load, transform and create dataloader
# https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=transforms+normalize#torchvision.transforms.Normalize
transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))
    ]   # (mean, std)
)

dataset = datasets.MNIST(root="../data", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# load optimizer
# adam work well with GAN
opti_gen = optim.Adam(gen.parameters(), lr=lr)
opti_disc = optim.Adam(disc.parameters(), lr=lr)
# loss function
loss = nn.BCELoss()

# log
fake_writer = SummaryWriter(f"runs/MNIST_GAN/fake")
real_writer = SummaryWriter(f"runs/MNIST_GAN/real")
step = 0

# Trainingloop
# https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html?highlight=tensor+zero_grad
# https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html?highlight=backward#torch.Tensor.backward
# https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html?highlight=step#torch.optim.Optimizer.step
for epoch in range(num_epoch):
    for batch_idx, (real_img, _) in enumerate(loader):  # (images, label)
        real_img = real_img.view(-1, 784).to(device)    # want to keep the batch -1
        batch_size = real_img.shape[0]

        # Discriminator
        # max log(D(real_img)) + log(1-D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)   # gaussian distribution
        fake_img = gen(noise)
        # D(real_img)
        disc_real_img = disc(real_img).view(-1)             # flatten
        # log(D(real_img))
        lossDis_real = loss(disc_real_img, torch.ones_like(disc_real_img))
        # D(G(z))
        disc_fake_img = disc(fake_img).view(-1)
        # log(1-D(G(z))
        lossDis_fake = loss(disc_fake_img, torch.zeros_like(disc_fake_img))
        # whole loss
        lossDis = (lossDis_real + lossDis_fake) / 2
        disc.zero_grad()
        lossDis.backward(retain_graph=True)
        opti_disc.step()

        # Generator
        # min log(1-D(G(z))) <-> max log(D(G(z)) -> simpler
        gen_output = disc(fake_img).view(-1)
        lossGen = loss(gen_output, torch.ones_like(gen_output))
        gen.zero_grad()
        lossGen.backward()
        opti_gen.step()

        # Setup for Tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epoch}] / "
                f"Loss D: {lossDis:.4f}, Loss G: {lossGen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fix_noise).reshape(-1, 1, 28, 28)
                data = real_img.reshape(-1, 1, 28, 28)
                fake_img_grid = torchvision.utils.make_grid(fake, normalize=True)
                real_img_grid = torchvision.utils.make_grid(data, normalize=True)

                fake_writer.add_image(
                    "Mnist Fake Images", fake_img_grid, global_step=step
                )

                real_writer.add_image(
                    "Mnist Real Images", real_img_grid, global_step=step
                )

                step += 1
