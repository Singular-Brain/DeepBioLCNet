import os

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder

import numpy as np


# Load MNIST data.
dataset = MNIST(
    None,
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=False,
    # transform=transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * intensity),
    #     transforms.CenterCrop(crop_size)]
)

test_dataset = MNIST(
    None,
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=False,
    train=False,
#     transform=transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Lambda(lambda x: x * intensity),
#         transforms.CenterCrop(crop_size)]
#     ),
)


target_classes = (1,)
mask = np.array([1 if dataset[i]['label'] in target_classes else 0 for i in range(len(dataset))])
mask_test = np.array([1 if test_dataset[i]['label'] in target_classes else 0 for i in range(len(test_dataset))])
np.savez(f'mask_{"_".join([str(i) for i in target_classes])}.npz', mask, mask_test)

