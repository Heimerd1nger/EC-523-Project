import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from util.util import accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def unlearning_finetuning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader,
    epochs = 1,
    repair_ep = 1):
    """Simple unlearning by finetuning."""
    print("Simple unlearning by finetuning")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()

    for ep in range(epochs):
        net.train()
        for sample in forget_loader:
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