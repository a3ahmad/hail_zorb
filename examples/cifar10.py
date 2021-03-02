import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from icecream import ic

import hail_zorb as hz

# DEBUGGING
ic.configureOutput(includeContext=True)

class BasicCifar10Classifier(hz.Module):
    def __init__(self):
        super().__init__()

        self.model = hz.Sequential(
            hz.Conv2d(3, 8, 3, padding=1),
            hz.ReLU(),
            hz.Conv2d(8, 8, 3, stride=2, padding=1),
            hz.ReLU(),

            hz.Conv2d(8, 16, 3, padding=1),
            hz.ReLU(),
            hz.Conv2d(16, 16, 3, stride=2, padding=1),
            hz.ReLU(),

            hz.Conv2d(16, 32, 3, padding=1),
            hz.ReLU(),
            hz.Conv2d(32, 32, 3, stride=2, padding=1),
            hz.ReLU(),

            hz.Conv2d(32, 64, 3, padding=1),
            hz.ReLU(),
            hz.Conv2d(64, 64, 3, padding=1),
            hz.ReLU(),

            hz.Flatten(),

            hz.Linear(1024, 256),
            hz.ReLU(),
            hz.Linear(256, 64),
            hz.ReLU(),
            hz.Linear(64, 10),

            hz.Softmax(dim=1),
        )


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/tvds/cifar-10',
                    help='Path of dataset')
parser.add_argument('--nonga', default=False,
                    action='store_true', help='Train a non-GA model')

args = parser.parse_args()


# data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_train = CIFAR10(args.path, train=True,
                        download=True, transform=transform)
cifar10_test = CIFAR10(args.path, train=False,
                       download=True, transform=transform)

train_loader = DataLoader(
    cifar10_train, batch_size=len(cifar10_train), num_workers=16)
train_input, train_labels = next(iter(train_loader))

test_loader = DataLoader(cifar10_test, batch_size=len(cifar10_test))
test_input, test_labels = next(iter(test_loader))

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BasicCifar10Classifier().to(device)

# training
hz.fit(model, train_input, train_labels)
