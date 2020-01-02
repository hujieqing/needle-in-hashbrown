#!/bin/bash

LR=0.001
LAYER_NUM=3
DATASET=communities
MODEL=GCN
LAMBDA1=1
LAMBDA2=0
WEIGHT_DECAY=0.0005
ALPHA=1
EPOCH_LOG=5
EPOCH_NUM=200
REPEAT_NUM=1

for LR in 0.01 0.001 0.0001;
do
  for WEIGHT_DECAY in 0.01 0.001 0.0001;
    do
      FULL_LOG_NAME=full_log.log
      PREFIX="${MODEL}_${LAYER_NUM}_${DATASET}_${LAMBDA1}_${LAMBDA2}_${ALPHA}_${LR}_${WEIGHT_DECAY}"
      TMP_DIR_NAME="tmp_${PREFIX}"
      mkdir $TMP_DIR_NAME
      python main.py --model $MODEL --layer_num $LAYER_NUM --dataset $DATASET --cpu --lambda1 $LAMBDA1 --lambda2 $LAMBDA2 --alpha $ALPHA --lr $LR --epoch_num $EPOCH_NUM --epoch_log $EPOCH_LOG --repeat_num $REPEAT_NUM --weight_decay $WEIGHT_DECAY >& $TMP_DIR_NAME/$FULL_LOG_NAME
      python analyzeLogs.py --filename $TMP_DIR_NAME/$FULL_LOG_NAME --prefix $PREFIX
      # rm -rf $tmpdirname
    done
done
