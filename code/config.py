
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='PyTorch code for UADAL')


parser.add_argument('--model', type=str, default='UAGRL',
                    help='')
parser.add_argument('--update_freq_D', type=int, default=1, metavar='S',
                    help='freq for D.')
parser.add_argument('--update_freq_G', type=int, default=1, metavar='S',
                    help='freq for G.')

parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                    help='resnet50, vgg')
parser.add_argument('--dataset', type=str, default='office31',
                    help='visda, office31, officehome')
parser.add_argument('--source_domain', type=str, default='A',
                    help='A, D, W ')
parser.add_argument('--target_domain', type=str, default='W',
                    help='A, D, W ')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--bottle_neck_dim', type=int, default=256, metavar='B',
                    help='bottle_neck_dim')

parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')
parser.add_argument('--exp_code', type=str, default='TEST',
                    help='experiment ID')
parser.add_argument('--set_gpu', type=int, default=0,
                    help='gpu setting 0 or 1')

parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate for main')
parser.add_argument('--w_lr', type=float, default=0.001, metavar='LR',
                    help='learning rate for warmup')
parser.add_argument('--glr_prop', type=float, default=0.1, metavar='LR',
                    help='g lr')

parser.add_argument('--lsr_eps', type=float, default=0.1, metavar='LR',
                    help='label smoothing')


parser.add_argument('--warmup_iter', type=int, default=200, metavar='S',
                    help='warmup iteration')
parser.add_argument('--main_iter', type=int, default=4000, metavar='S',
                    help='main iteration')
parser.add_argument('--report_term', type=int, default=100, metavar='S',
                    help='report term')
parser.add_argument('--update_term', type=int, default=200, metavar='S',
                    help='update term')

parser.add_argument('--scheduler', type=str, default='cos',
                    help='')


args = parser.parse_args()
