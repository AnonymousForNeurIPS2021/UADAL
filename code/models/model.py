from __future__ import print_function
import argparse
import time
import datetime
from utils import utils
from utils.utils import OptimWithSheduler

#from utils.networks import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss
from models.function import BetaMixture1D
from models.function import CrossEntropyLoss
from models.function import EntropyLoss
from models.basenet import *
import copy
from utils.utils import inverseDecayScheduler, CosineScheduler, StepScheduler
import math

class UAGRL():
    def __init__(self, args, num_class, dataset_train, dataset_test, dataset_train_ss):
        self.model = 'UADAL'  # Unknown-aware Domain adversarial learning
        self.args = args
        self.all_num_class = num_class
        self.known_num_class = num_class - 1  # except unknown
        self.dataset = args.dataset
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_train_ss = dataset_train_ss

        self.device = self.args.device#torch.device("cuda")

        self.build_warmup()

        self.ent_criterion = HLoss()

        self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)


    def build_warmup(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        self.G, self.E = utils.get_model_warmup(self.args, known_num_class=self.known_num_class)
        self.E.apply(weights_init_bias_zero)
        if self.args.cuda:
            self.G.to(self.args.device)
            self.E.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        self.opt_w_g = OptimWithSheduler(optim.SGD(self.G.parameters(), lr=self.args.glr_prop * self.args.w_lr, weight_decay=5e-4, momentum=0.9,
                           nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.w_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)


    def build_main(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        _, self.C, self.E, self.DC = utils.get_model_main(self.args, known_num_class=self.known_num_class,
                                                             all_num_class=self.all_num_class, domain_dim=3)

        self.E.apply(weights_init_bias_zero)
        self.DC.apply(weights_init_bias_zero)
        self.C.apply(weights_init_bias_zero)

        if self.args.cuda:
            self.E.to(self.args.device)
            self.C.to(self.args.device)
            self.DC.to(self.args.device)


        params_g = list(self.G.parameters())
        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.args.main_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.args.main_iter*self.args.update_freq_D)
        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.args.main_iter)


        self.opt_g = OptimWithSheduler(
            optim.SGD(params_g, lr=self.args.glr_prop * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.w_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_e)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)



    def network_initialization(self):
        if 'resnet' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()


    def warm_train(self):

        data_iter_s = iter(self.dataset_train_ss)
        len_train_source = len(self.dataset_train_ss)

        for step in range(self.args.warmup_iter+1):
            self.G.train()
            self.E.train()
            if self.args.cuda:
                if step%len_train_source==0:
                    data_iter_s = iter(self.dataset_train_ss)
                data = next(data_iter_s)
                img_s, label_s = data

                img_s = Variable(img_s.to(self.args.device))
                label_s = Variable(label_s.to(self.args.device))

            self.opt_w_g.zero_grad()
            self.opt_w_e.zero_grad()
            feat_s = self.G(img_s)
            out_s = self.E(feat_s)

            label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
            loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

            loss = loss_s
            loss.backward()
            self.opt_w_g.step()
            self.opt_w_e.step()
            self.opt_w_g.zero_grad()
            self.opt_w_e.zero_grad()


    def main_train(self):
        step = 0
        while step < self.args.main_iter + 1:
            for batch_idx, data in enumerate(self.dataset_train):
                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()
                alpha = float((float(2) / (1 + np.exp(-10 * float((float(step) / float(self.args.main_iter)))))) - 1)
                if self.args.cuda:
                    img_s, label_s = data['img_s'], data['label_s']
                    img_t, label_t = data['img_t'], data['label_t']
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))
                    img_t = Variable(img_t.to(self.args.device))
                if len(img_s) < self.args.batch_size:
                    break
                if len(img_t) < self.args.batch_size:
                    break
                step += 1
                if step >= self.args.main_iter + 1:
                    break
                #########################################################################################################
                out_t_free = self.E_freezed(self.G_freezed(img_t)).detach()
                w_unk_posterior = self.compute_probabilities_batch(out_t_free, 1)

                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(self.args.device)
                w_unk_posterior = w_unk_posterior.to(self.args.device)
                #########################################################################################################
                for d_step in range(self.args.update_freq_D):
                    self.opt_dc.zero_grad()
                    feat_s = self.G(img_s).detach()
                    out_ds = self.DC(feat_s)
                    label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
                    label_ds = nn.functional.one_hot(label_ds, num_classes=3)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))  # self.criterion(out_ds, label_ds)
                    label_dt_known = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=3)
                    label_dt_unknown = 2 * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt_unknown = nn.functional.one_hot(label_dt_unknown, num_classes=3)
                    feat_t = self.G(img_t).detach()
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_D = 0.5* (loss_ds + loss_dt)
                    loss_D.backward()
                    self.opt_dc.step()
                    self.opt_dc.zero_grad()
                #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()
                    feat_s = self.G(img_s)
                    out_ds = self.DC(feat_s)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))
                    feat_t = self.G(img_t)
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_G = alpha * (- loss_ds - loss_dt)
                    #########################################################################################################
                    out_Es = self.E(feat_s)
                    label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                    loss_cls_Es = CrossEntropyLoss(label=label_s_onehot,
                                                   predict_prob=F.softmax(out_Es, dim=1))  # self.criterion(out_s, label_s)
                    #########################################################################################################
                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))
                    #########################################################################################################
                    label_unknown = (self.known_num_class) * Variable(
                        torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)

                    label_unknown_lsr = label_unknown * (1 - self.args.lsr_eps)
                    label_unknown_lsr = label_unknown_lsr + self.args.lsr_eps / (self.known_num_class + 1)

                    out_Ct = self.C(feat_t)
                    loss_cls_Ctu = alpha * CrossEntropyLoss(label=label_unknown_lsr, predict_prob=F.softmax(out_Ct, dim=1),
                                                    instance_level_weight=w_unk_posterior)
                    loss_ent_Ctk = alpha * EntropyLoss(F.softmax(out_Ct, dim=1),
                                               instance_level_weight=w_k_posterior)


                    #########################################################################################################

                    loss = 0.5 * loss_G + loss_cls_Es + loss_cls_Cs + loss_ent_Ctk + 0.5*loss_cls_Ctu
                    loss.backward()
                    #########################################################################################################
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()
                    #########################################################################################################

                if step % (self.args.report_term) == 0:
                    C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos = self.test()
                    print('(%4s/%4s) |OS:%.4f |OS*:%.4f |UNK:%.4f |HOS:%.4f |'%(step, self.args.main_iter,C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos))
                    self.G.train()
                    self.C.train()
                    self.DC.train()
                    self.E.train()
                if step % self.args.update_term == 0:
                    if (step > 1) :
                        self.update_bmm_model()
                        self.freeze_GE()
                        self.network_initialization()

    def update_bmm_model(self):
        self.G.eval()
        self.E.eval()
        all_ent_t= torch.Tensor([])
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_test):
                if self.args.cuda:
                    img_t = data[0]
                    img_t = Variable(img_t.to(self.args.device))
                feat = self.G(img_t)
                out_t = self.E(feat)
                ent_t = self.ent_criterion(out_t)
                all_ent_t = torch.cat((all_ent_t, ent_t.cpu()))

        entropy_list = all_ent_t.data.numpy()
        loss_tr_t = (entropy_list - self.bmm_model_minLoss.data.cpu().numpy()) / (
                self.bmm_model_maxLoss.data.cpu().numpy() - self.bmm_model_minLoss.data.cpu().numpy() + 1e-6)

        loss_tr_t[loss_tr_t >= 1] = 1 - 10e-4  #1.0
        loss_tr_t[loss_tr_t <= 0] = 10e-4   # 0.0

        self.bmm_model = BetaMixture1D(max_iters=10)
        self.bmm_model.fit(loss_tr_t)
        self.bmm_model.create_lookup(1)

    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1] = 1.0
        batch_ent_t[batch_ent_t <= 0] = 0.0
        B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        B = torch.FloatTensor(B)
        return B

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)


    def test(self):
        self.G.eval()
        self.C.eval()

        total_pred = np.array([])
        total_label_v2 = np.array([])

        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_test):
                if self.args.cuda:
                    img_t, label_t = data[0], data[1]
                    img_t, label_t = Variable(img_t.to(self.args.device)), Variable(label_t.to(self.args.device))
                feat = self.G(img_t)
                out_t = F.softmax(self.C(feat), dim=1)
                pred = out_t.data.max(1)[1]

                pred_numpy = pred.cpu().numpy()
                total_pred = np.append(total_pred, pred_numpy)

                label_t_numpy = label_t.cpu().numpy()
                total_label_v2 = np.append(total_label_v2, label_t_numpy)

        max_target_label = int(np.max(total_label_v2)+1)
        m = utils.extended_confusion_matrix(total_label_v2, total_pred, true_labels=list(range(max_target_label)), pred_labels=list(range(self.all_num_class)))
        cm = m
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
        acc_os_star = sum([cm[i][i] for i in range(self.known_num_class)]) / self.known_num_class
        acc_unknown = sum([cm[i][self.known_num_class] for i in range(self.known_num_class, int(np.max(total_label_v2)+1))]) / (max_target_label - self.known_num_class)
        acc_os = (acc_os_star * (self.known_num_class) + acc_unknown) / (self.known_num_class+1)
        acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)
        return acc_os, acc_os_star, acc_unknown, acc_hos

