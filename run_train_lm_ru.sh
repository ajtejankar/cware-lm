#!/usr/bin/env bash

set -e
set -x

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ud_ru \
#     --word \
#     --save output/lm_1_word_small_ud_ru

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ud_ru \
#     --word \
#     --arch_large \
#     --save output/lm_2_word_large_ud_ru

# CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#     --data ./data/ud_ru \
#     --save output/lm_3_char_small_ud_ru

CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
    --data ./data/ud_ru \
    --arch_large \
    --save output/lm_4_char_large_ud_ru

