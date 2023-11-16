from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import grad
import numpy as np
import scipy.sparse as sp
device = "cuda" if torch.cuda.is_available() else "cpu"

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data
        curr += length

def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 2000
    o_vec = None
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for idx, data in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data, label = data
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def Wfisher(retain_loader,forget_loader, model, alpha=0.2,args=None):
    criterion = nn.CrossEntropyLoss()
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    retain_loader_b1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False
    )
    
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()

    for i, sample in enumerate(tqdm(forget_loader)):
        model.zero_grad()
        data, label = sample
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num
    total_2 = 0
    for i, data in enumerate(tqdm(retain_loader)):
        model.zero_grad()
        data, label = sample
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2
    perturb = woodfisher(
            model,
            retain_loader_b1,
            device=device,
            criterion=criterion,
            v=forget_grad - retain_grad,
)

    apply_perturb(model, args.alpha * perturb)

    return model
