from matbench.bench import MatbenchBenchmark
from preprocessing import *
import random
import json
import os
random.seed(0)

################################ Darwin Installation ################################

'''
Please clone and install the Darwin package

  1) git clone https://github.com/MasterAI-EAM/Darwin.git
  2) pip install -r requirements.txt
  3) download the base Darwin model from https://aigreendynamics-my.sharepoint.com/:f:/g/personal/yuwei_greendynamics_com_au/EvZEghuFSZZCguWrCsbk2QMB_eYqv-BRMM4VLhcK8TT4Zw?e=9bnqWW
     Our base model is built upon LLaMA-7b, trained with 9 datasets: Chembl, ESOL, MoosaviCp, MoosaviDiversity, NagasawaOPV, OPV, Pei, WaterStability
'''

mb = MatbenchBenchmark(
    autoload=True,
    subset=[
        "matbench_expt_is_metal",
        "matbench_steels",
        "matbench_glass",
        "matbench_expt_gap"
    ],
)

data_dir = 'train_test_data'
os.makedirs(data_dir)
os.makedirs('matbench_model')
fold_data = {0:[],1:[],2:[],3:[],4:[]}

for task in mb.tasks:
    task.load()
    task_name = task.dataset_name
    for fold in task.folds:
        # prepare the data for Darwin
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs,test_outputs = task.get_test_data(fold,include_target=True)
        # trainsform data into natural language
        if (task.dataset_name == 'matbench_expt_gap'):
            training_data = convert_gap(train_inputs,train_outputs)
            test_data = convert_gap(test_inputs,train=False)
        if (task.dataset_name == 'matbench_expt_is_metal'):
            training_data = convert_metal(train_inputs,train_outputs)
            test_data = convert_metal(test_inputs,train=False)
        if (task.dataset_name == 'matbench_steels'):
            training_data = convert_steels(train_inputs,train_outputs)
            test_data = convert_steels(test_inputs,train=False)
        if (task.dataset_name == 'matbench_glass'):
            training_data = convert_glass(train_inputs,train_outputs)
            test_data = convert_glass(test_inputs,train=False)
        # mix 4 tasks into a single training set
        fold_data[fold]+=training_data

        # create test dataset
        with open(data_dir +'/matbench_base_fold_'+str(fold)+'_'+task.dataset_name+'_test.json','w') as f:
            json.dump(test_data,f)

# creating the training dataset, training and evaluating the model
for fold in fold_data:
    
    # creating training dataset
    training_data = fold_data[fold]
    random.shuffle(training_data)
    data_path = data_dir +'/matbench_base_fold_'+str(fold)+'_train.json'
    output_path = 'matbench_model/fold'+str(fold)
    with open(data_path,'w') as f:
        json.dump(training_data,f)
    # train the model
    os.system("torchrun  --nproc_per_node=8 --master_port=1212 train.py \
    --model_name_or_path base_model \
    --data_path" + data_path + " \
    --bf16 True \
    --output_dir" + output_path + " \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False")
    # evaluate the model
    
    for task in mb.tasks:
        test_inputs,test_outputs = task.get_test_data(fold,include_target=True)
        test_data_path = data_dir +'/matbench_base_fold_'+str(fold)+'_'+task.dataset_name+'_test.json'
        os.system(f"python evaluate_matbench.py \
        --model_path {output_path} \
        --data_path {data_dir}/matbench_base_fold_{str(fold)}_{task.dataset_name}_test.json \
        --dataset {task.dataset_name} \
        --fold {fold}")
        
        # load the prediction result
        with open('matbench_base_fold_'+str(fold)+'_'+task.dataset_name+'_test_result.json') as f:
            data = json.load(f)
        transformed_data = {}
        for item in data:
            transformed_data[item['input'].strip()] = item['output']
        predicted_output = []
        for i in range(len(test_inputs)):
            predicted_output.append(transformed_data[test_inputs[i]])
        task.record(fold,predicted_output)

# save the result     
mb.to_file("results.json.gz")