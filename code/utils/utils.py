import torch.optim as opt
from models.basenet_v3 import *
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader.mydataset import ImageFolder, ImageFolder_ss
import os
def calculate_skewness(alpha, beta):
    skn = (2*(beta-alpha)*np.sqrt(alpha+beta+1))/((alpha+beta+2)*np.sqrt(alpha*beta))
    return skn


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

def bring_add_name(model_specs):
    add_name = ''
    for m_spc in model_specs:
        add_name += '%s_'%m_spc
    add_name = add_name[:-1]
    return add_name

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    #print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims #if isinstance(optims, Iterable) else [optims]
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def get_model_warmup(args, unk_num_class, all_num_class, domain_dim = 3, unit_size=100):
    net = args.net
    if net == 'vgg':
        model_g = VGGFc(device=args.device, model_name='vgg')# VGGBase_v5()
        model_e = ResNet_CLS(model_g.output_num(), unk_num_class, bottle_neck_dim=args.bottle_neck_dim)
    elif 'resnet' in net:
        model_g = ResNetFc(model_name='resnet50')#, model_path='pytorch_model/resnet50.pth')
        model_e = ResNet_CLS(model_g.output_num(), unk_num_class, bottle_neck_dim=args.bottle_neck_dim)
    return model_g, model_e


def get_model_main_v2(args, unk_num_class, all_num_class, domain_dim = 3, unit_size=100):
    net = args.net
    if net == 'vgg':
        model_g = VGGFc(device=args.device, model_name='vgg')# VGGBase_v5()
        model_c = ResNet_CLS_C(model_g.output_num(), all_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_e = ResNet_CLS(model_g.output_num(), unk_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_dc = ResNet_DC(model_g.output_num(), domain_dim, bottle_neck_dim=args.bottle_neck_dim)
    elif 'resnet' in net:
        model_g =ResNetFc(model_name='resnet50')#, model_path='pytorch_model/resnet50.pth')
        model_c = ResNet_CLS_C(model_g.output_num(), all_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_e = ResNet_CLS(model_g.output_num(), unk_num_class, bottle_neck_dim=args.bottle_neck_dim)
        model_dc = ResNet_DC(model_g.output_num(), out_dim=domain_dim, bottle_neck_dim=args.bottle_neck_dim)
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


def get_optimizer_warmup(G, E, discriminator_ps, center_loss, args):
    if args.net == 'vgg':
        for name, param in G.named_parameters():
            words = name.split('.')
            if words[1] == 'classifier':
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif args.net == 'resnet50':
        for name, param in G.named_parameters():
            words = name.split('.')
            if words[1] == 'layer4':
                param.requires_grad = True
            else:
                param.requires_grad = False

    params_ps = list(discriminator_ps.parameters())
    if args.center_lambda > 0:
        params_ps = params_ps + list(center_loss.parameters())

    if args.w_optimizer == 'sgd':
        print('Warmup would be sgd')
        opt_g2 = optim.SGD(G.parameters(), lr=args.glr_prop*args.w_lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        opt_e2 = optim.SGD(E.parameters(), lr=args.w_lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        opt_discriminator_ps2 = optim.SGD(params_ps, lr=args.w_lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif args.w_optimizer =='adam':
        print('Warmup would be adam')
        opt_g2 = optim.Adam(G.parameters(), lr=args.glr_prop*args.w_lr, weight_decay=5e-4)
        opt_e2 = optim.Adam(E.parameters(), lr=args.w_lr, weight_decay=5e-4,)
        opt_discriminator_ps2 = optim.Adam(params_ps, lr=args.w_lr, weight_decay=5e-4)
    return opt_e2, opt_g2, opt_discriminator_ps2

def bring_logger(results_log_dir, level='info'):
    import logging
    log1 = logging.getLogger('model specific logger')
    streamH = logging.StreamHandler()
    log1.addHandler(streamH)
    fileH = logging.FileHandler(results_log_dir)
    log1.addHandler(fileH)
    if level =='debug':
        log1.setLevel(level=logging.DEBUG)
    else:
        log1.setLevel(level=logging.INFO)
    return log1

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



def evaluate(total_pred, total_label, num_class):
    assert len(total_label) == len(total_pred)

    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)

    correct = np.sum(total_pred == total_label)
    size = len(total_pred)

    for t in range(num_class):
        t_ind = np.where(total_label == t)
        correct_ind = np.where(total_pred[t_ind[0]] == t)
        per_class_correct[t] += float(len(correct_ind[0]))
        per_class_num[t] += float(len(t_ind[0]))
    per_class_acc = per_class_correct / per_class_num

    os, os_star = float(per_class_acc.mean()), float(per_class_acc[:-1].mean())
    all, unk = float(float(correct) / float(size)), float(per_class_acc[-1].mean())

    return os, os_star, all, unk, per_class_acc


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

