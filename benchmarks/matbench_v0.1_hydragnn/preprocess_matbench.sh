echo "Processing matbench_$1"
for fold in {0..4}; do
  echo "Processing fold $fold"
  python preprocess_data.py --task_name matbench_$1 --fold $fold --modelname matbench_$1_${fold}
done
