#!/bin/bash
# User needs to provide the following 3 arguments
# 1.  cuda device index (i.e., 0, 1, ...), 
# 2. dataset name (i.e., email, communities, ppi), 
# 3. model name (GCN, SAGE)

# For baseline model sweep, set L2 = 0.0 and don't use hash_concat or hash_overwrite
# For hash model sweep, set hash_concat or hash_overwrite
# For Combined sweep, set L2 to a list of values with space delimited

CUDA=$1
DATASET=$2
MODEL=$3
COMMENT=$4
LEARNING_RATE=$5

LAYER_NUM=3
ALPHA=1
EPOCH_LOG=50
EPOCH_NUM=2000
REPEAT_NUM=2
for task in link_pair link;
do
  python main2.py --model $MODEL --task $task --layer_num $LAYER_NUM \
  --dataset $DATASET --gpu GPU --cuda $CUDA  \
    --alpha $ALPHA  --epoch_num $EPOCH_NUM \
    --epoch_log $EPOCH_LOG --repeat_num $REPEAT_NUM  \
      --early_stopping True --comment $COMMENT\
      --alpha $ALPHA \
      --l1 '1.0'\
      --l2 '0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0' \
      --lr $LEARNING_RATE \
      --weight_decay '0.1 0.01 0.001 0.0001'
done
