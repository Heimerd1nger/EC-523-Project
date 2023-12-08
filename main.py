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

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from util.util import get_starter_dataset,calculate_accuracy
from util.util import calculate_accuracy
from train_checkpoint import get_starter_dataset

from unlearner.SCRUB import unlearning_SCRUB

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


def run(**kwargs):
    args = arg_parser.parse_args()
    print(args)

    # writer = SummaryWriter()
    retain_loader, forget_loader, val_loader,test_loader,_ = get_starter_dataset()
    n_class = 10
    net = resnet18(weights=None, num_classes=n_class)
    weights_pretrained = torch.load('checkpoints/pre-train-model_epoch_40_lr_0.1_momentum_0.9_weightdecay_0.0005.pth', map_location=DEVICE)
    net.load_state_dict(weights_pretrained)
    net.to(DEVICE)
    model_s,acc_rs,acc_fs,acc_vs = unlearning_SCRUB(net,retain_loader,forget_loader,val_loader,is_starter=True,args=args)
    print(f'acc on retain {acc_rs}')
    print(f'acc on valid {acc_vs}')
    print(f'acc on forget {acc_fs}')
    # plot 
    indices = list(range(0,len(acc_rs)))
    plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c',alpha=1, label='validation-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.xlabel('ep',size=14)
    plt.ylabel('acc',size=14)
    plt.grid()
    plt.savefig(f"plots/SCRUB-epoch-{args.sgda_epochs}-Temp-{args.kd_T}-lr-{args.sgda_learning_rate}-bn-{args.sub_sample}.png")
    plt.show()
    
    # MIA
    # forget_losses = compute_losses(net, forget_loader)
    # test_losses = compute_losses(net, test_loader)
    # np.random.shuffle(forget_losses)
    # forget_losses = forget_losses[: len(test_losses)]
    # assert len(test_losses) == len(forget_losses)
    # samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    # labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    # mia_scores = simple_mia(samples_mia, labels_mia)
    # print(
    #     f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
    # )

if __name__ == '__main__':

    # experiment_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory to store logs if it doesn't exist
    results_dir = "results"
    plots_dir = "plots"
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
    