for i in {0..4}; do
  python -u finetune.py --datasetname matbench_"$1"_"$i" --modelname matbench_"$1"_"$i" --num_epochs $2 --checkpoint_dir --matbench
done
