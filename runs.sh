#!/bin/bash

# For training
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/maxmin_celeba_sn_resnet_sinkhorn_gan.yaml --train
# CUDA_VISIBLE_DEVICES=1 python main.py -c configs/minmax_celeba_sn_resnet_sinkhorn_gan.yaml --train

#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_resnet/cats_sn_resnet_sinkhorn_gan.yaml --train
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_resnet/celeba_sn_resnet_sinkhorn_gan.yaml --train
