import torch
import torch.nn as nn
import random
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from utils import to_one_hot, mixup_process, get_lambda


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class F3(nn.Module):
    def __init__(self):
        super(F3, self).__init__()

        self.f3 = nn.Sequential(OrderedDict([
            ('f3', nn.Linear(400, 120)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f3(img.reshape(img.shape[0], -1))
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self, num_classes):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, num_classes)),
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class lenet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, num_classes):
        super(lenet5, self).__init__()

        self.num_classes = num_classes
        self.c1 = C1()
        self.c2 = C2()
        self.f3 = F3()
        self.f4 = F4()
        self.f5 = F5(num_classes)

    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, noise=0.0):
        if mixup_hidden:
            layer_mix = random.randint(0, 4)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)

        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes, noise=noise)

        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.c1(out)
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.c2(out)
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.f3(out)
        if layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.f4(out)
        if layer_mix == 4:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.f5(out)

        if target is not None:
            return out, target_reweighted
        else:
            return out