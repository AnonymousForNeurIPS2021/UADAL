#! /bin/bash


python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain A --target_domain W --lr 0.0001 > 'office31_vgg_A_W.txt'
python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain A --target_domain D --lr 0.0001 > 'office31_vgg_A_D.txt'
python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain D --target_domain A --lr 0.0001 > 'office31_vgg_D_A.txt'
python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain D --target_domain W --lr 0.0001 > 'office31_vgg_D_W.txt'
python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain W --target_domain A --lr 0.0001 > 'office31_vgg_W_A.txt'
python3.6 train.py --dataset office31 --net vgg --warmup_iter 100 --main_iter 3000 --source_domain W --target_domain D --lr 0.0001 > 'office31_vgg_W_D.txt'
