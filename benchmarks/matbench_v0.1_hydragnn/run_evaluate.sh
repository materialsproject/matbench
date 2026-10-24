
python get_data.py
tar -zxvf pretrained_models.tar.gz
tar -zxvf finetuned_models.tar.gz

tasks=""
for task in "jdft2d" "mp_is_metal"; do
  bash preprocess_matbench.sh $task
  bash evaluate_our_finetune.sh $task
  tasks+=" matbench_$task"
done

bash compile.sh $tasks
