# train autoencoder for regression with no compression in the bottleneck layer
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb

# baseline in performance with support vector regression model
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, mean_pinball_loss


import sys
import pickle
import joblib
import re
import json


import pygad

import gzip
import shutil

import os



from matbench.bench import MatbenchBenchmark

from pystacknet.pystacknet import StackNetRegressor
from matminer.datasets import load_dataset
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.element import TMetalFraction
from matminer.featurizers.composition.element import Stoichiometry

from matminer.featurizers.composition.composite import Meredig
from matminer.featurizers.composition.element import BandCenter

from matminer.featurizers.conversions import StrToComposition

from matminer.datasets import get_all_dataset_info

import warnings
warnings.filterwarnings("ignore")

def model_test(models, model_names, X_train, y_train, X_test, y_test, norm=True):
    #utilizing for-loop to quickly analyze all 5 models
    mae = {}
    rmse = {}
    r2 = {}
    for model, name in zip(models, model_names):
        if norm:
            model = Pipeline([('scaler', RobustScaler()), ('model', model)])
        else:
            model = Pipeline([('model', model)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae[name] = mean_absolute_error(y_test, y_pred)
        rmse[name] = mean_squared_error(y_test, y_pred, squared=False)
        r2[name] = r2_score(y_test, y_pred)
        # print(f'{name} \n  R-Squared: {r2:.5f} \n  MAE: {mae:.5f} \n  RMSE: {rmse:.5f}')
    return mae, rmse, r2


def cross_val_test(model, dataX, dataY, model_type='regression', n_cv=10):

    # Розбиття для подальшої крос-валідації
    cv = KFold(n_splits=n_cv, shuffle=True)
    # Визначення набору метрик
    if model_type == 'regression':
        scorer = {'r2':make_scorer(r2_score),
                'mae': make_scorer(mean_absolute_error),
                'mse': make_scorer(mean_squared_error),
                'mape': make_scorer(mean_absolute_percentage_error)}
    if model_type == 'classification':
        pass

    # Оцінка якості моделі на різних наборах даних
    scores = cross_validate(model, dataX, dataY, scoring=scorer, cv=cv, return_train_score=True)
    return scores

def remove_zero_lists(list_of_lists):
    """Видаляє всі списки, що складаються виключно з нулів, зі списку списків."""
    return [sublist for sublist in list_of_lists if not all(x == 0 for x in sublist)]


def split_list(input_list, chunk_size):
    """Розділити список на підсписки заданої довжини."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def find_indices_of_ones(binary_list):
    """Повертає список індексів, на позиціях яких у бінарному списку стоять одиниці."""
    return [index for index, value in enumerate(binary_list) if value == 1]

def select_models_by_indices(ind, models):
    """Генерує список підсписків, де кожен підсписок відповідає індексам з `ind` для списку `models`."""
    return [[models[index] for index in sublist] for sublist in ind]

# # sol - приклад хромосоми, з якої генерується модель
# sol = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# split_list(sol, 7)

# gen_stacknet(sol)



def plot_scatter(y_true, y_pred):
    """
    Функція для побудови точкової діаграми справжніх значень та прогнозованих значень.

    :param y_true: список або масив справжніх значень
    :param y_pred: список або масив прогнозованих значень
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', marker='o')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Лінія ідеального прогнозу
    plt.title('Точкова діаграма: справжні значення vs. прогнозовані значення')
    plt.xlabel('Тестові значення (y_test)')
    plt.ylabel('Прогнозовані значення (y_pred)')
    plt.grid(True)
    plt.show()


# sys.path.insert(0,'../..')

def unzip(path_to_gz_file, output_path):
# Відкриваємо gz файл
    with gzip.open(path_to_gz_file, 'rb') as f_in:
        # Відкриваємо вихідний файл для запису
        with open(output_path, 'wb') as f_out:
            # Копіюємо вміст gz файлу в вихідний файл
            shutil.copyfileobj(f_in, f_out)

    print("Файл успішно розархівовано.")
    
    

def delete_file(file_path):
    """Видаляє файл, якщо він існує."""
    try:
        os.remove(file_path)
        print(f"Файл {file_path} успішно видалено.")
    except FileNotFoundError:
        print(f"Файл {file_path} не знайдено.")
    except PermissionError:
        print(f"Недостатньо прав для видалення файлу {file_path}.")
    except Exception as e:
        print(f"Помилка при видаленні файлу: {e}")



def process_json_files(directory):

    unzip('./results.json.gz', './results.json')
    results = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)

                # Extracting the MAE scores from each fold
                mae_scores = []
                for fold in range(5):
                    key = f"fold_{fold}"
                    mae_scores.append(data["tasks"]["matbench_steels"]["results"][key]["scores"]["mae"])

                # Calculating mean and standard deviation
                mae_mean = np.mean(mae_scores)
                mae_std = np.std(mae_scores)

                results.append({
                    "filename": filename,
                    "mae_scores": mae_scores,
                    "mean": mae_mean,
                    "std": mae_std
                })

    # delete_file('./results.json')
    # delete_file('./results.json.gz')

    return results


def data_preproc(df_steels, compos_name):


    stc = StrToComposition()
    df_steels = stc.featurize_dataframe(df_steels, compos_name, pbar=False)

    ef = ElementFraction()
    tm = TMetalFraction()
    st = Stoichiometry()
    meredig = Meredig()
    bc = BandCenter()

    df_steels_bc = bc.featurize_dataframe(df_steels, "composition")
    df_steels_ef = ef.featurize_dataframe(df_steels, "composition")

    df_steels_ef = df_steels_ef.loc[:, (df_steels_ef == 0).mean() <= 0.6]
    df_steels_ef

    df_steels_tm = tm.featurize_dataframe(df_steels, "composition")

    df_steels_st = st.featurize_dataframe(df_steels, "composition")

    # df_steels_meredig = meredig.featurize_dataframe(df_steels, "composition")

    data_steel = df_steels_ef.drop([compos_name, 'composition'], axis=1)

    df = pd.concat([data_steel,
                        df_steels_st['0-norm'], df_steels_st['2-norm'],
                        df_steels_st['3-norm'], df_steels_st['5-norm'],
                        df_steels_st['7-norm'], df_steels_st['10-norm'],
                        df_steels_tm['transition metal fraction'],
                        df_steels_bc['band center']
                ], axis=1)

    return df

def gen_stacknet(sol):

    models = [RandomForestRegressor(random_state=42),
        LinearRegression(),
        Ridge(random_state=42),
        XGBRegressor(random_state=42),
        HistGradientBoostingRegressor(random_state=42),
        AdaBoostRegressor(random_state=42),
        GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000,
                                    subsample=1.0, criterion='friedman_mse',
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_depth=5, random_state=42),

          ]

    len_model = len(models)
    restacking = bool(sol[-2])
    retraining = bool(sol[-1])
    layers = sol[0:-2]
    sublayers = split_list(layers, len_model)
    sublayers = remove_zero_lists(sublayers)
    if sublayers:
        models_by_layers = [find_indices_of_ones(sublayers[i]) for i in range(len(sublayers))]
        selected_models = select_models_by_indices(models_by_layers, models)
        selected_models.append([selected_models[-1][-1]])

    else:
        selected_models = models[0]

    stack_net = StackNetRegressor(selected_models, metric="r2", folds=5,
    restacking=restacking, use_retraining=retraining,
    random_state=42, n_jobs=16, verbose=0)

    return stack_net


def model_eval(model):

    mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])

    for task in mb.tasks:
        task.load()
        for fold in task.folds:

            # Inputs are either chemical compositions as strings
            # or crystal structures as pymatgen.Structure objects.
            # Outputs are either floats (regression tasks) or bools (classification tasks)
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            df_steels_mb = pd.DataFrame({'compos':list(train_inputs), 'target':list(train_outputs)})

            df = data_preproc(df_steels_mb, 'compos')

            X_train = df.drop('target', axis=1)
            y_train = df['target']


            # train and validate your model
            model.fit(X_train,y_train)

            # Get testing data
            test_inputs = task.get_test_data(fold, include_target=False)

            test_df = pd.DataFrame({'compos':list(test_inputs)})
            X_test = data_preproc(test_df, 'compos')

            # Predict on the testing data
            # Your output should be a pandas series, numpy array, or python iterable
            # where the array elements are floats or bools
            # predictions = my_model.predict(test_inputs)
            predictions = model.predict(X_test)

            predictions = predictions.flatten()

            # Record your data!
            task.record(fold, predictions)




    # Save your results
    mb.to_file("results.json.gz")
    
    return 0
    
    


def fitness_eval(results_dic, lambd=0.0):

    fitness = 1 / (results_dic['mean'] + lambd*results_dic['std'] ) * 1000
    return fitness


sol_dict = []

def fitness_func(ga_instance, solution, solution_idx):
    global sol_dict 
    model = gen_stacknet(solution)
    model_eval(model)
    directory = "./"  # Replace with the path to your directory with json files
    results = process_json_files(directory)
    fitness = fitness_eval(results[0])
    print('------------sol----------\n', solution)
    print('------------sol_idx----------\n', solution_idx)
    print('------------fit----------\n', fitness)
    sol_dict.append([solution, solution_idx, fitness, results[0]['mean'], results[0]['std']])
    return fitness



# 5 layers for initial solution
# init_solution = [0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]

# 5 layers for initial solution
# len([0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0])
# init_solution = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

init_solution = [1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0]


def genetic_instance():
    global init_population
    global init_solution
    ga_instance = pygad.GA(num_generations=10,
                            num_parents_mating=2,
                            fitness_func=fitness_func,
                            initial_population=init_population,
                            # sol_per_pop=int(sol_per_pop),
                            num_genes=len(init_solution),
                            # on_start=on_start,
                            # on_fitness=on_fitness,
                            # on_parents=on_parents,
                            # on_crossover=on_crossover,
                            # on_mutation=on_mutation,
                            # on_generation=on_generation,
                            crossover_probability= 60/100 + 0.1,
                            mutation_probability= 80/100 + 0.1, # [0,1]
                            mutation_by_replacement=False, # True
                            mutation_percent_genes = 80, # [0,100]
                            # on_stop=on_stop
                            save_solutions=True,
                            crossover_type='uniform',  # 83.88 13 robust
                            mutation_type='swap',  

                            # crossover_type='scattered',  
                            # mutation_type='swap',  

                            # crossover_type='uniform',  # 
                            # mutation_type='inversion',  
                            )

    ga_instance.run()

    return ga_instance


import random

def generate_random_binary_list(length):
    """Генерує випадковий бінарний список заданої довжини."""
    return [random.randint(0, 1) for _ in range(length)]


def set_init_population(n_pop):
    global init_solution

    init_pop = [init_solution]
    for i in range(n_pop):
        new_sol = generate_random_binary_list(len(init_solution))
        init_pop.append(new_sol)

    # self.init_population = init_pop
    return init_pop



def ga_results(ga_instance):
    ga_instance.plot_genes(color='blue')
    ga_instance.plot_fitness(color='blue')
    ga_instance.plot_new_solution_rate(color='blue')



init_population = set_init_population(4)


ga = genetic_instance()



solution, solution_fitness, solution_idx = ga.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

best_sol = solution

model = gen_stacknet(best_sol)
model_eval(model)
directory = "./"  # Replace with the path to your directory with json files
results = process_json_files(directory)

ga_results(ga)


# len([0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0])
# ([0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0.,
#        1., 0., 0., 1., 1., 0.]) 83.88 13 robust

# [0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
#        0., 0., 1., 1., 1., 0.])


model = gen_stacknet([0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0.,1., 0., 0., 1., 1., 0.])

model_eval(model)
directory = "./"  # Replace with the path to your directory with json files
results = process_json_files(directory)

# fitness_func(ga, [0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0], 0)


