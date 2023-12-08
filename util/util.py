import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model, model_selection
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import TensorDataset

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchvision.models import resnet18
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




def label_distribution(data_loader):
    class_counts = Counter()
    for _, labels in data_loader:
        class_counts.update( [label.item() for label in labels])
    return class_counts

# to ensure reproducible results across runs
def subloader_kmeans(net,forget_loader,cluster=2):
    batch_size = 128
    label_list = []
    sample_list = []
    for inputs, labels in forget_loader:
        label_list.extend(labels.tolist())
        sample_list.extend(inputs.tolist())
    forget_inputs = torch.tensor(sample_list).cuda()
    forget_targets = torch.tensor(label_list).cuda()
    assert len(forget_inputs) == len(sample_list), "Tensor length is not equal"
    forget_outputs = net(forget_inputs)
    kmeans = KMeans(n_clusters=cluster, random_state=32)
    cluster_labels = kmeans.fit_predict(forget_outputs.to('cpu').detach())
    return cluster_labels    


def subloader(net,forget_loader,exp=False,mode=None):
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
      if exp:
          return easy_loader,hard_loader,easy_set,diff_set
      return easy_loader,hard_loader
 



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
    # train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

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
    
    return retain_loader, forget_loader, val_loader, test_loader

def acc_eval(net, retain_loader, forget_loader, test_loader):
    """Return accuracy on a dataset given by the data loaders."""
    return accuracy(net, retain_loader), accuracy(net, forget_loader), accuracy(net, test_loader)

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

def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def calculate_accuracy(net, dataloader,mode=True):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in dataloader:
            if mode:
                inputs, targets = sample
            else:
                inputs = sample["image"]
                targets = sample["age_group"]
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def standard_mia(net,forget_loader,test_loader,n_splits=10, random_state=0):
    forget_losses = compute_losses(net, forget_loader)
    test_losses = compute_losses(net, test_loader)
    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(test_losses)]
    assert len(test_losses) == len(forget_losses)    
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_scores = simple_mia(samples_mia, labels_mia)
    return mia_scores.mean()

###########################################SCRUB############################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy(output, target):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res[0]
    
def accuracy(model_outputs, targets):

    _, predicted = torch.max(model_outputs.data, 1)
    
    # Count the number of correct predictions
    correct_predictions = (predicted == targets).sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / targets.size(0) # Assuming targets is not batched
    return accuracy

######################################## INFLUENCE FUNCTION ###############################################
# computes average gradient of the full dataset
def compute_full_grad(model, device, data_loader, loss_fn, lam=0):
    full_grad = None
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        loss_with_reg(model, data, target, loss_fn, lam)
        grad = params_to_vec(model.parameters(), grad=True)
        if full_grad is None:
            full_grad = grad * data.size(0) / len(data_loader.dataset)
        else:
            full_grad += grad * data.size(0) / len(data_loader.dataset)
        model.zero_grad()
    param_vec = params_to_vec(model.parameters())
    return full_grad, param_vec

def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()

def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param

def batch_grads_to_vec(parameters):
    N = parameters[0].shape[0]
    vec = []
    for param in parameters:
        vec.append(param.view(N,-1))
    return torch.cat(vec, dim=1)

def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(param.size(0), -1).size(1)
        grads.append(vec[:, :size].view_as(param))
        vec = vec[:, size:]
    return grads




######################################## Visualization ###############################################
def plot_confusion_matrix(model, dataloader, figsize=(10, 7), cmap='Blues', font='Arial', fontsize=12):
    """
    Plots the confusion matrix for the given model and dataloader.

    Parameters:
    - model: The pre-trained model to evaluate.
    - dataloader: DataLoader containing the dataset to evaluate the model on.
    - figsize: Tuple representing the figure size (width, height).
    - cmap: Color map for the confusion matrix.
    - font: Font family for the plot text.
    - fontsize: Font size for labels and title.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    true_labels = []
    predictions = []

    # Iterate over the dataloader and collect predictions
    with torch.no_grad():  # Turn off gradients for evaluation
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.tolist())
            predictions.extend(predicted.tolist())

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plotting
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)  # Scale for seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    
    # Set the font
    plt.rc('font', family=font)

    # Labeling the plot
    plt.xlabel('Predicted Labels', fontsize=fontsize)
    plt.ylabel('True Labels', fontsize=fontsize)
    plt.title('Confusion Matrix', fontsize=fontsize + 2)
    plt.show()