from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from torch.nn import Module, Linear
from torch.nn.functional import relu, softmax
from torch.utils.data import DataLoader

transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_dataloader(train=True):
    trainset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=train, num_workers=2)

    return trainloader


class BasicModel(Module):
    def __init__(self):
        super().__init__()

        self.fc1 = Linear(in_features=32 * 32 * 3, out_features=200)
        self.fc2 = Linear(in_features=200, out_features=150)
        self.fc3 = Linear(in_features=150, out_features=10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = softmax(self.fc3(x))

        return x


def predict(net, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicated = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    return correct / total
