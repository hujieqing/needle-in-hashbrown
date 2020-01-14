#!/bin/bash
# User needs to provide the following 3 arguments
# 1.cuda device index (i.e., 0, 1, ...), 2. dataset name (i.e., email, communities, ppi), 3. model name (GCN, SAGE)
CUDA=$1
DATASET=$2
MODEL=$3

LAYER_NUM=3
#LAMBDA1=1
#LAMBDA2=0
WEIGHT_DECAY=0.01
ALPHA=1
EPOCH_LOG=5
EPOCH_NUM=50
REPEAT_NUM=5
TMP_DIR_NAME="sweep_results/tmp_${MODEL}_${DATASET}_hash"

rm -rf $TMP_DIR_NAME
mkdir -p $TMP_DIR_NAME

for LAMBDA1 in 0.0 0.1 1 10;
  do
  for LAMBDA2 in 0.0 0.1 1 10;
    do			      
    for LR in 0.01 0.001 0.0001;
      do
      for WEIGHT_DECAY in 0.01 0.001 0.0001;
        do
	PREFIX="${LAYER_NUM}_${DATASET}_L1_${LAMBDA1}_L2_${LAMBDA2}_A${ALPHA}_LR${LR}_WD${WEIGHT_DECAY}"
	LOG_FILE_NAME=`echo $PREFIX | sed 's/\./d/g'`
	LOG_FILE_PATH="${TMP_DIR_NAME}/${LOG_FILE_NAME}"
	SUMMARY_PATH="${TMP_DIR_NAME}/summary"
        python main.py --model $MODEL --layer_num $LAYER_NUM --dataset $DATASET --gpu GPU --cuda $CUDA --lambda1 $LAMBDA1 --lambda2 $LAMBDA2 --alpha $ALPHA --lr $LR --epoch_num $EPOCH_NUM --epoch_log $EPOCH_LOG --repeat_num $REPEAT_NUM --weight_decay $WEIGHT_DECAY --early_stopping True --hash_concat |tee -a $LOG_FILE_PATH
        python analyzeLogs.py --filename $LOG_FILE_PATH --prefix $PREFIX | tee -a $SUMMARY_PATH
        done
    done
  done
done

