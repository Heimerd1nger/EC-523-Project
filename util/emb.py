import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from util.util import get_starter_dataset,subloader,compute_losses,simple_mia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_feature(loader,_extractor):
    ret = []
    y = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = _extractor(inputs).squeeze().cpu()
            # outputs = nn.functional.normalize(outputs)
            ret.append(outputs)
            y.append(targets.cpu())
    ret = torch.tensor(np.concatenate(ret))
    y = np.concatenate(y)
    ret = F.normalize(ret, dim=-1)
    # mean = ret.mean(dim=1, keepdim=True)
    # std = ret.std(dim=1, keepdim=True)
    # ret =  (ret - mean) / std
    # ret = np.linalg.norm(ret, axis=-1, keepdims=True)
    return ret, y

def extract_data_by_target(dataloader, target_value):
    data_list = []  # List to store data with the matching target value

    # Iterate over the DataLoader
    for data, targets in dataloader:
        # Select data that matches the target value
        mask = targets == target_value
        filtered_data = data[mask]
        
        # Move the data to CPU and convert to numpy if it's not already
        filtered_data = filtered_data.cpu().numpy()
        
        # Append the filtered data to the list
        data_list.append(filtered_data)
    
    # Concatenate all collected data into a single numpy array
    concatenated_data = np.concatenate(data_list, axis=0)
    return torch.tensor(concatenated_data)


def emb(loader1,loader2,target):
    emb1 = extract_data_by_target(loader1,target).to(DEVICE)
    emb2 = extract_data_by_target(loader2,target).to(DEVICE)
    d_cos = F.cosine_similarity(emb1, emb2, dim=1)
    d_euclidien = F.cosine_similarity(emb1, emb2, dim=1)
    return d_cos, d_euclidien