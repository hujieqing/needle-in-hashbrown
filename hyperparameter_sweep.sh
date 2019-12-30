#!/bin/bash

LR=0.001
LAYER_NUM=2
DATASET=communities
MODEL=GCN
LAMBDA1=1
LAMBDA2=0
ALPHA=1
EPOCH_LOG=2
REPEAT_NUM=2

FULL_LOG_NAME=full_log.log
TMP_DIR_NAME="tmp_${MODEL}_${LAYER_NUM}_${DATASET}_${LAMBDA1}_${LAMBDA2}_${ALPHA}_${LR}"
mkdir $TMP_DIR_NAME
python main.py --model $MODEL --layer_num $LAYER_NUM --dataset $DATASET --cpu --lambda1 $LAMBDA1 --lambda2 $LAMBDA2 --alpha $ALPHA --lr $LR --epoch_num 5 --epoch_log $EPOCH_LOG --repeat_num $REPEAT_NUM >& $TMP_DIR_NAME/$FULL_LOG_NAME
python analyzeLogs.py --filename $TMP_DIR_NAME/$FULL_LOG_NAME --prefix "${MODEL}_${LAYER_NUM}_${DATASET}_${LAMBDA1}_${LAMBDA2}_${ALPHA}_${LR}"
# rm -rf $tmpdirname
