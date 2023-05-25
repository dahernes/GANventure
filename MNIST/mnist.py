# Download and quickcheck of the mnist dataset
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (1.0,))]
)

batch_size = 16

# load dataset from torchvision
mnist_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

loader = torch.utils.data.DataLoader(
    mnist_set,
    batch_size=batch_size,
    shuffle=True
)


# show image function
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get random data
dataiter = iter(loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print datapoints / image shapes
print('Num Datapoints in Dataset: {}'.format(len(mnist_set)))
print('Image Attributes: {}'.format(images[0].shape))
