import numpy as np
import torch
import copy
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn.functional as F
from util.util import calculate_accuracy, AverageMeter, accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


# def adjust_learning_rate(epoch, opt, optimizer):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
#     new_lr = opt.sgda_learning_rate
#     if steps > 0:
#         new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lr
#     return new_lr
    

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
def validate(val_loader, model, criterion, opt, quiet=False):
    """validation"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
    return top1.avg, losses.avg


def train_distill(is_starter, epoch, train_loader, module_list, criterion_list, optimizer, opt, split, quiet=False,swa_model = None):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top1_f =  AverageMeter()
    num_batches = len(train_loader)


    for idx, data in enumerate(train_loader):
        ## subsampling
        if  opt.sub_sample!=0.0 and split == "minimize":
            if (idx+1) == int(num_batches*opt.sub_sample):
                print(idx+1)
                break


        if is_starter:
            inputs, targets = data
        else:
            inputs = data["image"]
            targets = data["age_group"] 
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        # ===================forward=====================
        logit_s = model_s(inputs)

        with torch.no_grad():
            logit_t = model_t(inputs)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, targets)
        loss_div = criterion_div(logit_s, logit_t)
    

        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, opt.smoothing)

        if split == "minimize" and not quiet:
            acc1 = accuracy(logit_s, targets)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
        elif split == "maximize" and not quiet:
            accf = accuracy(logit_s, targets)
            kd_losses.update(loss.item(), inputs.size(0))
            top1_f.update(accf, inputs.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
    
    if split == "minimize":
        return top1.avg, losses.avg
    else:
        return top1_f.avg, kd_losses.avg


def unlearning_SCRUB(net, retain, forget, validation, is_starter,args):

    """Unlearning by SCRUB.

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
    acc_rs = []
    acc_fs = []
    acc_vs = []
    checkpoints = None 
    diff = 1.0
    model_t = copy.deepcopy(net)
    model_s = net
    

    # swa
    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
        1 - beta) * averaged_model_parameter + beta * model_parameter
    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss


    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          momentum=args.sgda_momentum,
                          weight_decay=args.sgda_weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.sgda_epochs)

    #teacher
    module_list.append(model_t)
    
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()
    
    for epoch in range(1, args.sgda_epochs + 1):
        # lr = adjust_learning_rate(epoch, args, optimizer)
        acc_r, loss_r = validate(retain, model_s, criterion_cls, args, True)
        acc_f, loss_f = validate(forget, model_s, criterion_cls, args, True)
        acc_v, loss_v = validate(validation, model_s, criterion_cls, args, True)

        acc_rs.append(acc_r)
        acc_fs.append(acc_f)
        acc_vs.append(acc_v)
        if epoch <= args.msteps:
            maximize_loss = 0
            forget_acc = 0
        train_acc, train_loss = 0, 0

        if epoch <= args.msteps:
            forget_acc, maximize_loss = train_distill(is_starter, epoch, forget, module_list, criterion_list, optimizer, args, "maximize",swa_model=swa_model)

        train_acc, train_loss = train_distill(is_starter, epoch, retain, module_list, criterion_list, optimizer, args, "minimize",swa_model=swa_model)
        scheduler.step()

        # REWIND
        if args.checkpoints:
            if abs(acc_f-acc_v) < diff:
                checkpoints = copy.deepcopy(model_s.state_dict())
                diff = abs(acc_f-acc_v)
                print(diff)

        # if epoch >= args.sstart:
        #     swa_model.update_parameters(model_s)

        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {:.2f}\t forget_acc: {:.2f}".format(maximize_loss, train_loss, train_acc, forget_acc))
    acc_r, _ = validate(retain, model_s, criterion_cls, args, True)
    acc_f, _ = validate(forget, model_s, criterion_cls, args, True)
    acc_v, _ = validate(validation, model_s, criterion_cls, args, True)
    acc_rs.append(acc_r)
    acc_fs.append(acc_f)
    acc_vs.append(acc_v)
    torch.save(checkpoints,f"checkpoints/scrub/scrub-model_epoch_{args.sgda_epochs}_lr_{args.sgda_learning_rate}_temp_{args.kd_T}_subsamp_{args.sub_sample}.pth")
    model_s.load_state_dict(checkpoints)
    model_s.eval()
    return model_s,acc_rs,acc_fs,acc_vs
