# matbench_v0.1: Darwin

### Algorithm description: 

Fine-tuning DARWIN Natural Science Large Language Model

#### Notes:
We provide prompts and call-and-return of our model. The code for evaluating the benchmarks is available at https://github.com/MasterAI-EAM/Darwin-SIT, our base model is available at https://aigreendynamics-my.sharepoint.com/:f:/g/personal/yuwei_greendynamics_com_au/EvZEghuFSZZCguWrCsbk2QMB_eYqv-BRMM4VLhcK8TT4Zw?e=9bnqWW. To train our model, it requires at least 4*A100(80G)

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_darwin).

### References (in bibtex format): 

```
('@misc{xie2023large,\n'
 ' title={Large Language Models as Master Key: Unlocking the Secrets of '
 'Materials Science with GPT},\n'
 ' author={Tong Xie and Yuwei Wan and Wei Huang and Yufei Zhou and Yixuan Liu '
 'and Qingyuan Linghu and Shaozhou Wang and Chunyu Kit and Clara Grazian and '
 'Wenjie Zhang and Bram Hoex},\n'
 ' year={2023},\n'
 ' eprint={2304.02213},\n'
 ' archivePrefix={arXiv},\n'
 ' primaryClass={cs.CL}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 4/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✓ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['git+https://github.com/MasterAI-EAM/Darwin.git',
            'matbench==0.1.0',
            'numpy',
            'rouge_score',
            'fire',
            'openai',
            'transformers>=4.28.1',
            'torch',
            'sentencepiece',
            'tokenizers>=0.13.3',
            'wandb']}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2723| 0.6679| 0.3996| 7.6000 |
 | fold_1 | 0.2873| 0.6843| 0.3629| 6.4000 |
 | fold_2 | 0.2945| 0.7388| 0.4516| 6.3900 |
 | fold_3 | 0.2836| 0.7482| 0.4408| 6.8000 |
 | fold_4 | 0.2949| 0.7535| 0.4748| 6.8000 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2865 | 0.2949 | 0.2723 | 0.0083 |
| rmse | 0.7185 | 0.7535 | 0.6679 | 0.0354 |
| mape* | 0.4259 | 0.4748 | 0.3629 | 0.0398 |
| max_error | 6.7980 | 7.6000 | 6.3900 | 0.4400 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_expt_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9635| 0.9634| 0.9630| 0.9634 |
 | fold_1 | 0.9634| 0.9634| 0.9633| 0.9634 |
 | fold_2 | 0.9543| 0.9543| 0.9548| 0.9543 |
 | fold_3 | 0.9624| 0.9624| 0.9626| 0.9624 |
 | fold_4 | 0.9553| 0.9553| 0.9558| 0.9553 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9598 | 0.9635 | 0.9543 | 0.0041 |
| balanced_accuracy | 0.9598 | 0.9634 | 0.9543 | 0.0041 |
| f1 | 0.9599 | 0.9633 | 0.9548 | 0.0038 |
| rocauc | 0.9598 | 0.9634 | 0.9543 | 0.0041 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_glass`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8160| 0.7607| 0.8733| 0.7607 |
 | fold_1 | 0.8002| 0.7450| 0.8617| 0.7450 |
 | fold_2 | 0.8169| 0.7694| 0.8725| 0.7694 |
 | fold_3 | 0.8275| 0.7840| 0.8796| 0.7840 |
 | fold_4 | 0.8195| 0.7749| 0.8740| 0.7749 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8160 | 0.8275 | 0.8002 | 0.0089 |
| balanced_accuracy | 0.7668 | 0.7840 | 0.7450 | 0.0133 |
| f1 | 0.8722 | 0.8796 | 0.8617 | 0.0058 |
| rocauc | 0.7668 | 0.7840 | 0.7450 | 0.0133 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 129.1587| 182.3484| 0.0884| 565.5000 |
 | fold_1 | 108.8460| 144.9490| 0.0799| 399.7000 |
 | fold_2 | 122.0081| 176.7949| 0.0864| 469.0000 |
 | fold_3 | 146.6677| 200.0602| 0.1076| 577.3000 |
 | fold_4 | 109.7855| 165.5786| 0.0813| 628.9000 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 123.2932 | 146.6677 | 108.8460 | 13.9542 |
| rmse | 173.9462 | 200.0602 | 144.9490 | 18.2839 |
| mape* | 0.0887 | 0.1076 | 0.0799 | 0.0099 |
| max_error | 528.0800 | 628.9000 | 399.7000 | 82.4129 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




