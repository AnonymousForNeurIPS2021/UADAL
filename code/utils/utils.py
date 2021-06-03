import torch.optim as opt
from models.basenet_v3 import *
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import os

def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def CosineScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    cos = (1 + np.cos((step / max_iter) * np.pi)) / 2
    lr = initial_lr * cos
    return lr
def StepScheduler(step, initial_lr, gamma=500, power=0.2, max_iter=1000):
    divide = step // 500
    lr = initial_lr * (0.2 ** divide)
    return lr

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    return NGPU

def get_model_warmup(args, known_num_class):
    net = args.net
    if net == 'vgg':
        model_g = VGGFc(device=args.device, model_name='vgg')
        model_e = ResNet_CLS(model_g.output_num(), known_num_class)
    elif 'resnet' in net:
        model_g = ResNetFc(model_name='resnet50')
        model_e = ResNet_CLS(model_g.output_num(), known_num_class)
    return model_g, model_e


def get_model_main(args, known_num_class, all_num_class, domain_dim = 3):
    net = args.net
    if net == 'vgg':
        model_g = VGGFc(device=args.device, model_name='vgg')
        model_c = ResNet_CLS_C(model_g.output_num(), all_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_e = ResNet_CLS(model_g.output_num(), known_num_class)
        model_dc = ResNet_DC(model_g.output_num(), out_dim=domain_dim)
    elif 'resnet' in net:
        model_g =ResNetFc(model_name='resnet50')
        model_c = ResNet_CLS_C(model_g.output_num(), all_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_e = ResNet_CLS(model_g.output_num(), known_num_class)
        model_dc = ResNet_DC(model_g.output_num(), out_dim=domain_dim)
    return model_g, model_c, model_e, model_dc


class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


def bring_data_transformation(source_data, target_data, evaluation_data, dataset, args=None):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix

