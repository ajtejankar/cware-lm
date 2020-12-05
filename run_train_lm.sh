#!/usr/bin/env bash

set -e
set -x

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ptb \
#     --word \
#     --save output/lm_1_word_small_ptb

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ptb \
#     --word \
#     --arch_large \
#     --save output/lm_2_word_large_ptb

CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
    --data ./data/ptb \
    --save output/lm_6_char_small_ptb

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ptb \
#     --arch_large \
#     --save output/lm_4_char_large_ptb

