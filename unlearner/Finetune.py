import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from util.util import accuracy
from util.util import calculate_accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def unlearning_finetuning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader,
    args,
    ):
    """Simple unlearning by finetuning."""
    acc_rs = []
    acc_fs = []
    acc_vs = []

    print("Simple unlearning by finetuning")
    epochs = args.sgda_epochs
    lr = args.sgda_learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()
    for ep in range(epochs):
        net.train()
        for sample in retain_loader:
            inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc_r = calculate_accuracy(net,retain_loader)
        acc_f = calculate_accuracy(net,forget_loader)
        acc_v = calculate_accuracy(net,val_loader)
        print('Epoch: {}  Reatin acc: {} Forget acc: {} Validation acc: {}'.format(ep,acc_r,acc_f,acc_v))
        acc_rs.append(acc_r)
        acc_fs.append(acc_f)
        acc_vs.append(acc_v)

    if args.checkpoints:
        torch.save(net.state_dict(),f"checkpoints/finetuning/finetuning-model_epoch_{args.sgda_epochs}_lr_{args.sgda_learning_rate}.pth")
    net.eval()
    return net,acc_rs,acc_fs,acc_vs