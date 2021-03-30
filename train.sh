#!/usr/bin/env bash
PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python3 train_collages.py \
--dataset_name ' ' \
--model_name 'model_collages_all_resnet50L_noWeight_moreAlb_flips_250x400_2' \
