epochs=(100 10)

python get_data.py
tar -zxvf pretrained_models.tar.gz

num=0
tasks=""
for task in "jdft2d" "mp_is_metal"; do
  bash preprocess_matbench.sh $task
  bash finetune.sh $task ${epochs[num]}
  let num++
  bash evaluate_your_finetune.sh $task
  tasks+=" matbench_$task"
done

bash compile.sh $tasks
