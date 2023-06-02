import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_data_transformers():
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])
    
    return train_transforms, test_transforms


def load_mnist(train_transforms, test_transforms):
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return train_data, test_data

def create_loader(**kwargs):
    test_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    return train_loader, test_loader


def plot(batch_data, batch_label, num_rows, num_cols):
    fig = plt.figure()
    for i in range(num_cols):
        plt.subplot(num_rows,num_cols,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def plot_metrics(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
