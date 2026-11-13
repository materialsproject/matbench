# AlchemBERT

#### Introduction
We aim to evaluate the performance of BERT in predicting matbench results.

#### Instructions

1.  Use str_matbench_data_generation.py to generate data, where material properties are described in natural language. To save runtime, the pre-generated data is available in the folders data_cif_v1 and data_nl_v1.
2.  Run the scheduling.sh script (without providing a path) to automatically start training tasks, with one task assigned to each available GPU. Alternatively, you can run train.py individually to train a specific task.
3.  After training, use print_log.py to read the log files and print the results of each task.
4.  Finally, run scheduling.sh again (this time providing the model storage path) to generate prediction.csv files for all tasks. Then, use matbench_submit.py to create files ready for matbench submission. Note: You can use the following command to simplify JSON files and verify that the results to be submitted match the ones you computed.
```

jq 'walk(if type == "object" then del(.data) else . end)' my_loaded_predictions_benchmark.json > output.json
```
