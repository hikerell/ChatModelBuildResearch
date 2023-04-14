#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --data_path c-s-ale/alpaca-gpt4-data \
   --data_split 10,0,0 \
   --model_name_or_path Hikerell/shine-RLHF-20230414-on-opt-1.3b \
   --gradient_accumulation_steps 2 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
