#!/bin/bash
# User needs to provide the following 3 arguments
# 1.  cuda device index (i.e., 0, 1, ...), 
# 2. dataset name (i.e., email, communities, ppi), 
# 3. model name (GCN, SAGE)

# For baseline model sweep, set L2 = 0.0 and don't use hash_concat or hash_overwrite
# For hash model sweep, set hash_concat or hash_overwrite
# For Combined sweep, set L2 to a list of values with space delimited

DATASET=$1
MODEL=$2

LAYER_NUM=2
ALPHA=1
EPOCH_LOG=100
EPOCH_NUM=2000
REPEAT_NUM=2
for task in link_pair link;
do
  python main2.py --model $MODEL --task $task --layer_num $LAYER_NUM \
  --dataset $DATASET --cpu  \
    --alpha $ALPHA  --epoch_num $EPOCH_NUM \
    --epoch_log $EPOCH_LOG --repeat_num $REPEAT_NUM  \
      --early_stopping True --comment PGNNFCombined \
      --alpha $ALPHA --approximate 2 \
      --l1 '0.0 0.1 1 10' \
      --l2 '0.0 0.1 1 10' \
      --lr '0.1 0.01 0.001 0.0001' \
      --weight_decay '0.01 0.001 0.0001'
done
     

