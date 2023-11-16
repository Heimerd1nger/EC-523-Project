
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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from util.util import get_starter_dataset,calculate_accuracy,compute_losses,simple_mia
from unlearner.random_label import unlearning_rdn
from unlearner.relabel import unlearning_rlb
from unlearner.benchmark import unlearning_finetuning
from unlearner.negative_gradient import unlearning_ng


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run():
    retain_loader, forget_loader, val_loader,test_loader = get_starter_dataset(128)
    n_class = 10
    net = resnet18(weights=None, num_classes=n_class)
    weights_pretrained = torch.load('weights_resnet18_cifar10.pth', map_location=DEVICE)
    net.load_state_dict(weights_pretrained)
    net.to(DEVICE)
    unlearning_ng(net, retain_loader, forget_loader,val_loader,impair_ep=1)

    print('unlearn acc on retain is',calculate_accuracy(net,retain_loader))
    print('unlearn acc on forget is',calculate_accuracy(net,forget_loader))
    print('unlearn acc on valid is',calculate_accuracy(net,val_loader))

    forget_losses = compute_losses(net, forget_loader)
    test_losses = compute_losses(net, test_loader)
    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(test_losses)]
    assert len(test_losses) == len(forget_losses)
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_scores = simple_mia(samples_mia, labels_mia)
    print(
        f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
    )

if __name__ == '__main__':
    print(f'using {DEVICE} now')
    run()
    # experiment_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # Define a directory to store the logs
    # plot_dir = f"logs/{experiment_date_time}"
    # # Create a SummaryWriter instance
    # writer = SummaryWriter(log_dir=plot_dir)

    # # Create a directory to store logs if it doesn't exist
    # log_dir = "results/ng"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # # Generate the filename based on the current date
    # log_filename = os.path.join(log_dir, f"experiment_{experiment_date_time}.txt")
    # with open(log_filename, "w") as log_file:
    #     log_file.write(f"Experiment Date: {experiment_date_time}\n")
    #     log_file.write(f"Running on device:: {DEVICE.upper()}\n")
    #     run()
    # writer.close()