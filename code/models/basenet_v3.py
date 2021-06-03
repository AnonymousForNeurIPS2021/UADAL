import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from models.function import ReverseLayerF, WeightedForwardLayerF
import numpy as np
from torch.autograd.variable import *


class GradReverse(torch.autograd.Function):
    # def __init__(self, lambd):
    #     self.lambd = lambd
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        #return (grad_output * -self.lambd)
        return (grad_output.neg())
def grad_reverse(x):
    return GradReverse.apply(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc, self).__init__()

        self.model_resnet = resnet_dict[model_name](pretrained=True)

        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class VGGFc(BaseFeatureExtractor):
    def __init__(self, device, model_name='vgg19', normalize=True):
        super(VGGFc, self).__init__()

        self.model_vgg = models.vgg19(pretrained=True)
        self.normalize = normalize
        self.mean = False
        self.std = False
        model_vgg = self.model_vgg
        mod = list(model_vgg.features.children())
        self.features = nn.Sequential(*mod)
        mod2 = list(model_vgg.classifier.children())[:-1]
        self.classifier = nn.Sequential(*mod2)
        self.__in_features = 4096
        self.device = device

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).to(
                self.device)
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).to(
                self.device)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class ResNet_CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim):
        super(ResNet_CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,
        )
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class ResNet_CLS_C(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim):
        super(ResNet_CLS_C, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)

        self.main = nn.Sequential(self.bottleneck,
                                  nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                                self.fc))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class ResNet_DC(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim):
        super(ResNet_DC, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,
        )

    def forward(self, x, alpha=1.0, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, alpha)
        else:
            x = WeightedForwardLayerF.apply(x, alpha)
        for module in self.main.children():
            x = module(x)
        return x#, out

