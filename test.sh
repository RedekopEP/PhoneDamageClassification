#!/usr/bin/env bash
PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python3 test_collages.py \
--dataset_name ' ' \
--model_name '' \
