#! /bin/bash


python3.6 train.py --dataset office31 --net resnet50 --source_domain A --target_domain W --lr 0.00005 > 'office31_resnet_A_W.txt'
python3.6 train.py --dataset office31 --net resnet50 --source_domain A --target_domain D --lr 0.0005 > 'office31_resnet_A_D.txt'
python3.6 train.py --dataset office31 --net resnet50 --source_domain D --target_domain A --lr 0.0005 > 'office31_resnet_D_A.txt'
python3.6 train.py --dataset office31 --net resnet50 --source_domain D --target_domain W --lr 0.0005 > 'office31_resnet_D_W.txt'
python3.6 train.py --dataset office31 --net resnet50 --source_domain W --target_domain A --lr 0.0005 > 'office31_resnet_W_A.txt'
python3.6 train.py --dataset office31 --net resnet50 --source_domain W --target_domain D --lr 0.0005 > 'office31_resnet_W_D.txt'
