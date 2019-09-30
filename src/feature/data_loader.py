import torchvision
import torch


fashion_mnist_data = torchvision.datasets.FashionMNIST(
    './fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    download=True)

data_loader = torch.utils.data.DataLoader(
    dataset=fashion_mnist_data,
    batch_size=128,
    shuffle=True)

fashion_mnist_data_test = torchvision.datasets.FashionMNIST(
    './fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset=fashion_mnist_data_test,
    batch_size=128,
    shuffle=True)
