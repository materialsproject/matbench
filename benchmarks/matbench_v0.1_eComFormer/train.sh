#!/bin/sh

GPU=1

fold=0

CUDA_VISIBLE_DEVICES=${GPU} \
python ./train.py \
--output_dir="../matbench_mp_e_form_equivariant_max25_epoch500_lr1e-3_L1_fold"$fold \
--max_neighbors=25 \
--epochs=500 \
--batch_size=64 \
--task_name="matbench_mp_e_form" \
--lr=1e-3 \
--criterion='l1' \
--fold_num=$fold \
--multi_GPU \
--model_variant="matformerequivariant"
