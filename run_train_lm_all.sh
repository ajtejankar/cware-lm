#!/usr/bin/env bash

set -e
set -x

lang=$1

CUDA_VISIBLE_DEVICES=$2 python train_lm.py \
    --data ./data/${lang} \
    --word \
    --save output/lm_1_word_small_${lang} &

CUDA_VISIBLE_DEVICES=$3 python train_lm.py \
    --data ./data/${lang} \
    --word \
    --arch_large \
    --save output/lm_2_word_large_${lang} &

CUDA_VISIBLE_DEVICES=$4 python train_lm.py \
    --data ./data/${lang} \
    --save output/lm_3_char_small_${lang} &

CUDA_VISIBLE_DEVICES=$5 python train_lm.py \
    --data ./data/${lang} \
    --arch_large \
    --save output/lm_4_char_large_${lang} &


# for lang in {de,es,fr,ru}
# do
#     CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#         --data ./data/${lang} \
#         --word \
#         --save output/lm_1_word_small_${lang}

#     CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#         --data ./data/${lang} \
#         --word \
#         --arch_large \
#         --save output/lm_2_word_large_${lang}

#     CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#         --data ./data/${lang} \
#         --save output/lm_3_char_small_${lang}

#     CUDA_VISIBLE_DEVICES=$1 python train_lm.py \
#         --data ./data/${lang} \
#         --arch_large \
#         --save output/lm_4_char_large_${lang}
# done

