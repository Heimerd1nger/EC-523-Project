import os
import logging
import sys
import arg_parser
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from datetime import datetime

from torch.utils.data import DataLoader
from util.util import calculate_accuracy,compute_losses,simple_mia

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_starter_dataset(batch_size=256,bs_f=256,bs_r=256):
    '''Get the CIFAR-10 starter dataset'''
    RNG = torch.Generator().manual_seed(42)
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,generator=RNG)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2,generator=RNG)

    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=bs_f, shuffle=True, num_workers=2
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=bs_r, shuffle=True, num_workers=2, generator=RNG
    )
    
    return retain_loader, forget_loader, val_loader, test_loader, train_loader

def train_model(net, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        # Validation loop
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
        scheduler.step()
    net.eval()
    return net


if __name__ == '__main__':

    retain_loader, forget_loader, val_loader,test_loader,train_loader = get_starter_dataset()
    _ = [print(len(i.dataset) )for i in get_starter_dataset()]
    n_class = 10
    epochs = 100
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    # net = resnet18(weights=None, num_classes=n_class)
    # net.to(DEVICE)
    # net  = train_model(net, train_loader, val_loader, epochs)

    # checkpoint_path = f"checkpoints/pre-train-model_epoch_{epochs}_lr_{lr}_momentum_{momentum}_weightdecay_{weight_decay}.pth"
    # os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # torch.save(net.state_dict(), checkpoint_path)
    # print(f"Checkpoint performance, Retain_loader Accuracy: {calculate_accuracy(net,retain_loader):.2f}%, forget_loader Accuracy: {calculate_accuracy(net,forget_loader):.2f}%, Test_loader Accuracy: {calculate_accuracy(net,test_loader):.2f}%")

    # print(f"Pre-trained Checkpoint saved to {checkpoint_path}")
    retrain_net = resnet18(weights=None, num_classes=n_class)
    retrain_net.to(DEVICE)
    retrain_net = train_model(retrain_net,retain_loader, val_loader,epochs)
    checkpoint_path = f"checkpoints/retrain-model_epoch_{epochs}_lr_{lr}_momentum_{momentum}_weightdecay_{weight_decay}.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(retrain_net.state_dict(), checkpoint_path)
    print(f"Checkpoint performance, Retain_loader Accuracy: {calculate_accuracy(retrain_net,retain_loader):.2f}%, forget_loader Accuracy: {calculate_accuracy(retrain_net,forget_loader):.2f}%, Test_loader Accuracy: {calculate_accuracy(retrain_net,test_loader):.2f}%")

    print(f"Re-trained Checkpoint saved to {checkpoint_path}")
