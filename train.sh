#!/usr/bin/env bash

CONFIG=$1

#python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch
#torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 basicsr/train.py -opt ../options/train/train_restormer.yml --launcher pytorch
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 basicsr/train.py -opt $CONFIG --launcher pytorch
