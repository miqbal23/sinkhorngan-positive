#!/bin/bash
#source /opt/conda/etc/profile.d/conda.sh
#conda activate torch-gan
# For training
CUDA_VISIBLE_DEVICES=1 python main.py -c configs/maxmin_celeba_sn_resnet_sinkhorn_gan.yaml --train
CUDA_VISIBLE_DEVICES=1 python main.py -c configs/minmax_celeba_sn_resnet_sinkhorn_gan.yaml --train

#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_resnet/cats_sn_resnet_sinkhorn_gan.yaml --train
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_resnet/celeba_sn_resnet_sinkhorn_gan.yaml --train
#
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_dcgan/mnist_sn_dcgan_sinkhorn_gan.yaml --train
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_dcgan/mnist_sn_dcgan_sinkhorn_sgd.yaml --train
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/sn_dcgan/mnist_sn_dcgan_wgan.yaml --train
#
#CUDA_VISIBLE_DEVICES=1 python main.py -c configs/mlp/mnist_mlp_wgan_div.yaml --train
#CUDA_VISIBLE_DEVICES=0 python main.py -c configs/mlp/mnist_mlp_wgan_gp.yaml --train

# For Tensorboard
# see https://stackoverflow.com/a/52178270/3165035 for samples_per_plugin explanation
# tensorboard --logdir=../experiments/runs --samples_per_plugin "scalar=0"
