for i in {0..4}; do
  python -u evaluate.py --datasetname matbench_"$1"_"$i" --modelname matbench_"$1"_"$i" --checkpoint_root "$PWD"/finetuned_models/matbench_"$1"_"$i" --output_dir results_"$1" --matbench
done
