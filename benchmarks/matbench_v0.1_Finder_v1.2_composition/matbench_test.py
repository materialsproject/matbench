# ~ Composition-only Finder_v1.2 algorithm ~

import os
import shutil

import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from matbench.bench import MatbenchBenchmark

################################ Finder Installation ################################

'''
Please clone and install the Finder package at this location. You may also find installation instructions here --> https://github.com/ihalage/Finder

  1) git clone https://github.com/ihalage/Finder.git
  2) python -m venv Finder_env  # using a virtual environment
  3) source Finder_env/bin/activate
  4) cd Finder/
  5) pip install -r requirements.txt
  6) You may have to install the matbench package in the same virtual environment

'''


# Defining a number of helper function to prepare the data for the Finder algorithm. The data processing codes
# are adapted from CrabNet's submission to the matbench repository.

# condense_formula takes a material and returns the chemical formula in the correct format for Finder
def condense_formula(mat):
    if isinstance(mat, str):
        return mat, 'no structure'
    else:
        return mat.formula.replace(' ', ''), mat.to(fmt='cif')


# change_input runs condense_formula on all the input data used for training
def change_input(train_inputs):
  formula = []
  cif = []
  for inp in train_inputs:
    f, c = condense_formula(inp)
    formula.append(f)
    cif.append(c)
  return formula, cif

#make_df creates a data frame containing the train inputs and outputs for Finder
def make_df(formula, cif, train_outputs):
  input_df = pd.DataFrame({'formula': formula, 'cif': cif, 'target': train_outputs})
  return input_df

#make_df_test creates a data frame containing the test inputs for Finder
def make_df_test(formula, cif, test_outputs):
  test_df = pd.DataFrame({'formula': formula, 'cif': cif, 'target': test_outputs})
  # test_df['target'] = np.nan
  return test_df

#split_train_val splits the training data into two sets: training and validation
def split_train_val(df):
  df = df.sample(frac = 1.0, random_state = 12)
  val_df = df.sample(frac = 0.1, random_state = 12)
  train_df = df.drop(val_df.index)

  return train_df, val_df


#Defining a subset containing all of the regression tasks from the matbench tasks
subset = ['matbench_dielectric',
          'matbench_jdft2d',
          'matbench_log_gvrh',
          'matbench_log_kvrh',
          'matbench_mp_e_form',
          'matbench_mp_gap',
          'matbench_perovskites',
          'matbench_phonons']


mb = MatbenchBenchmark(autoload=False, subset=subset)
data_dir = 'matbench_data/matbench_temp'
os.makedirs(data_dir, exist_ok= True)

model_dir = 'matbench_models' # to save best models
results_dict = {}

for task in mb.tasks:
    print('\n\n================ NEW TASK =================')
    task.load()
    mat_prop = task.dataset_name
    os.makedirs(f'{data_dir}/{mat_prop}', exist_ok= True)
    os.makedirs(f'{model_dir}/{mat_prop}', exist_ok= True)  # to save models


    for i, fold in enumerate(task.folds):
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)


        #Preparing the inputs data for Finder
        formula, cif = change_input(train_inputs)
        df = make_df(formula, cif, train_outputs)

        #Creating the training and validation sets
        train_df, val_df = split_train_val(df)
        train_df.to_csv(f'{data_dir}/{mat_prop}/train.csv')
        val_df.to_csv(f'{data_dir}/{mat_prop}/val.csv')

        #Getting and preparing the testing data
        formula, cif = change_input(test_inputs)
        output_df = make_df_test(formula, cif, test_outputs)
        output_df.to_csv(f'{data_dir}/{mat_prop}/test.csv')

        ##### training and evaluating Finder models #####
        '''
        Please note that although the number of epochs is fixed at 500 in the following example run, some matbench
        tasks like mp_e_form require about 800 epochs to converge whereas smaller databases like phonons and dielectric
        converge within 300 epochs. One workaround is to define subsets variable above as a dictionary with specified
        hyperparameters for each task and feed those hyperparameters as arguments below.
        '''
        os.chdir('Finder/Finder') # change directory for convenience
        os.system('python trainer.py --train-path '+ f'../../{data_dir}/{mat_prop}/train.csv ' +
            '--val-path ' + f'../../{data_dir}/{mat_prop}/val.csv ' +
            '--test-path ' + f'../../{data_dir}/{mat_prop}/test.csv ' +
            '--epochs ' + '500' + ' --batch-size 128 --train --test --patience 300 --max-no-atoms 500')

        os.chdir('../..') # jump back

        # saving the best model for each fold for each task
        shutil.copytree('Finder/Finder/saved_models/best_model_gnn', f'{model_dir}/{mat_prop}/best_model_gnn_{i}', dirs_exist_ok=True)

        # extracting test results and record
        predictions = pd.read_csv('Finder/Finder/results/test_results.csv')['prediction'].tolist()
        task.record(fold, predictions)


# Saving our results
mb.to_file("results.json.gz")

