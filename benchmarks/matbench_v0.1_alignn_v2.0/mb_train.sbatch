#!/bin/bash
#SBATCH --job-name=mb_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=0
#SBATCH --output=mb_%j.out

# usage: sbatch mb_train.sbatch <tag>      e.g. sbatch mb_train.sbatch mp_gap_f0
# Trains one fold of one task. Expects, under $WORK, the dataset directory
# mb_<tag>/ and the per-fold config config_mb_<tag>.json, both written by
# make_matbench_dataset.py.
TAG=$1
WORK=/path/to/matbench_workdir      # edit: where datasets, configs and runs live
ALIGNN=/path/to/alignn              # edit: ALIGNN source checkout
PY=python                           # edit: python of the environment with torch

export PYTHONPATH=$ALIGNN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# per-run workdir: the graph caches (sample*_data) are written relative to the
# current directory, so concurrent jobs must never share a working directory
mkdir -p $WORK/work_${TAG}
cd $WORK/work_${TAG}

$PY -m alignn.train_alignn \
    --root_dir $WORK/mb_${TAG} \
    --config_name $WORK/config_mb_${TAG}.json \
    --output_dir $WORK/mb_run_${TAG}
