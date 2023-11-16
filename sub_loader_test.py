import os
import logging
import sys
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
from torch.utils.data.dataset import TensorDataset
from datetime import datetime
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from util.util import get_starter_dataset,calculate_accuracy,compute_losses,simple_mia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomStream:
    def __init__(self,filename,console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass 

def unlearning(net, retain, forget=None, validation=None,epochs = 1):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net

def unlearning_rnlabel(net, retain, forget=None, validation=None):

    """Unlearning by random label.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()
    for ep in range(epochs):
        net.train()
        for iter_num, sample in enumerate(forget):
            inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            rnd_label = torch.randint(0, 10, size=targets.size()).to(DEVICE)
            loss = criterion(outputs, rnd_label.detach())
            loss.backward()
            optimizer.step()

        for sample in retain:
            inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    scheduler.step()
    net.eval()
    return net

def unlearning_ga(net, retain, forget=None, validation=None,epochs=1):

    """Unlearning by gradient ascent.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()
    for ep in range(epochs):
        net.train()
        for iter_num, sample in enumerate(forget):
            inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = - criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        for sample in retain:
            inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    scheduler.step()
    net.eval()
    return net

def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def subloader(net,forget_loader):
      batch_size = 128
      label_list = []
      sample_list = []
      # Iterate through the data loader and collect the labels
      for inputs, labels in forget_loader:
          label_list.extend(labels.tolist())
          sample_list.extend(inputs.tolist())
      print("the size of the forget set is {}".format(len(sample_list)))
      # Load the pre-trained model

      forget_inputs = torch.tensor(sample_list).cuda()
      forget_targets = torch.tensor(label_list).cuda()
      assert len(forget_inputs) == len(sample_list), "Tensor length is not equal"

      forget_outputs = net(forget_inputs)

      topk_indices_values, _ = torch.topk(forget_outputs, 2, dim=1) # Get the top two confident class
      diff_confidence = abs(topk_indices_values[:,0] - topk_indices_values[:,1])
      _, topk_indices_diff = torch.topk(diff_confidence, 5000, largest=True)
      # print(len(topk_indices_diff))

      ranked_set = forget_inputs[topk_indices_diff]
      ranked_target = forget_targets[topk_indices_diff]
      diff_set = ranked_set[0:2500]
      diff_targets = ranked_target[0:2500]
      easy_set = ranked_set[2500:]
      easy_targets = ranked_target[2500:]
      easy_dataset = TensorDataset(easy_set, easy_targets)
      diff_dataset = TensorDataset(diff_set, diff_targets)
      easy_loader = DataLoader(easy_dataset, batch_size=batch_size, shuffle=False)  # You can set shuffle to True for randomization
      hard_loader = DataLoader(diff_dataset, batch_size=batch_size, shuffle=False)  
      return easy_loader,hard_loader

def run():
    print("experiment on ep - subloaders-easy(gradient ascent) performance")

    torch.manual_seed(42)
    n_class = 10

    retain_loader, forget_loader, val_loader,test_loader = get_starter_dataset(128,bs_f=128,bs_r=128)
    print("size of the forget lodaer is {}".format(len(forget_loader)))
    print("size of the retain lodaer is {}".format(len(retain_loader)))
    net = resnet18(weights=None, num_classes=n_class)
    weights_pretrained = torch.load('weights_resnet18_cifar10.pth', map_location=DEVICE)
    net.load_state_dict(weights_pretrained)
    net.to(DEVICE)
    easy_loader,hard_loader = subloader(net,forget_loader)
    acc_r = []
    acc_test = []
    acc_f = []
    acc_hard = []
    acc_easy = []
    for ep in range(1,18,2):
      print("num of unlearn epoch is {}".format(ep))
      unlearn_model = resnet18(weights=None, num_classes=n_class)
      unlearn_model.load_state_dict(net.state_dict())  
      unlearn_model.to(DEVICE)
      updated_model = unlearning_ga(unlearn_model,retain_loader,forget=hard_loader, epochs=ep)
    # updated_model = resnet18(weights=None, num_classes=n_class)
    # updated_model.load_state_dict(torch.load("retrain_weights_resnet18_cifar10.pth",map_location=DEVICE))
    # updated_model.to(DEVICE)
      acc_r.append(accuracy(updated_model, retain_loader))
      acc_test.append(accuracy(updated_model, test_loader))
      acc_f.append(accuracy(updated_model, forget_loader))
      acc_hard.append(accuracy(updated_model, hard_loader))
      acc_easy.append(accuracy(updated_model, easy_loader))

      print(f"Retain set accuracy: {100.0 * acc_r[-1]:0.1f}%")
      print(f"Test set accuracy: {100.0 *  acc_test[-1]:0.1f}%")
      print(f"Forget set accuracy: {100.0 *  acc_f[-1]:0.1f}%")
      print(f"Hard set accuracy: {100.0 *  acc_hard[-1]:0.1f}%")
      print(f"Easy set accuracy: {100.0 *  acc_easy[-1]:0.1f}%")
    print(acc_r)
    print(acc_test)
    print(acc_f)
    print(acc_hard)
    print(acc_easy)

if __name__ == '__main__':
    # experiment_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory to store logs if it doesn't exist
    results_dir = "results"
    # Get the current date and time
    current_datetime = datetime.now()
    # Create a subdirectory with the current date
    current_date_dir = os.path.join(results_dir, current_datetime.strftime("%Y-%m-%d"))
    # current_plot_dir = os.path.join(plots_dir, current_datetime.strftime("%Y-%m-%d"))
    os.makedirs(current_date_dir, exist_ok=True)
    # os.makedirs(current_plot_dir, exist_ok=True)

    # Create a log file with the exact time as the file name
    log_file_name = current_datetime.strftime("%H-%M-%S.log.txt")
    log_file_path = os.path.join(current_date_dir, log_file_name)
    

    # Configure the logging module to write to the log file
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,  # Adjust the log level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Redirect sys.stdout to the custom stream
    custom_stream = CustomStream(log_file_path, sys.stdout)
    sys.stdout = custom_stream
    # unlearn
    run()
    sys.stdout = sys.__stdout__
    
  