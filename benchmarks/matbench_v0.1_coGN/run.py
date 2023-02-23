import numpy as np
import h5py
import json

from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as ks

from matbench.bench import MatbenchBenchmark
from crystalgnns.crystal_preprocessing.crystal_preprocessors import (RadiusAsymmetricUnitCell,
                                                                     KNNAsymmetricUnitCell,
                                                                     VoronoiAsymmetricUnitCell)
from crystalgnns.graph_dataset_tools.graph_tuple import HDFGraphTuple, GraphTuple
from crystalgnns.graph_dataset_tools.model_preprocessors import GNPreprocessor
from crystalgnns.datasets.matbench_datasets import MatbenchDataset
from crystalgnns.models.graph_network.graph_network import get_model


def get_model_preprocessor(use_multiplicity = True, use_voronoi_area = False,
                           use_line_graph = True, line_graph_direction=[1,1]):
    model_inputs = [GNPreprocessor.Inputs.offset,
        GNPreprocessor.Inputs.atomic_number,
        GNPreprocessor.Inputs.edge_indices]
    if use_multiplicity:
        model_inputs.append(GNPreprocessor.Inputs.multiplicity)
    if use_voronoi_area:
        model_inputs.append(GNPreprocessor.Inputs.voronoi_ridge_area)
    if use_line_graph:
        model_inputs.append(GNPreprocessor.Inputs.line_graph_edge_indices)
    model_preprocessor = GNPreprocessor(inputs=model_inputs,
                                        line_graph_directions=line_graph_direction)
    return model_preprocessor

def get_lr_scheduler(dataset_size, batch_size, epochs, lr_start=0.0005, lr_stop=1e-5):
    steps_per_epoch = dataset_size / batch_size
    num_steps = epochs * steps_per_epoch
    scheduler = ks.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr_start,
                                                        decay_steps=num_steps,
                                                        end_learning_rate=lr_stop)
    return scheduler


def train_procedure(model_cfg, crystal_preprocessor, matbench_datasets_subset, use_multiplicity = True,
                    use_voronoi_area = False, use_line_graph = True, line_graph_direction = [1,1],
                    dataset_cache=Path('./dataset_cache'), results_cache=Path('./results'),
                    use_scaler=True, epochs=800, batch_size=64):
    # MatBench training procedure
    dataset_cache = Path(dataset_cache)
    results_cache = Path(results_cache)
    
    model_preprocessor = get_model_preprocessor(use_multiplicity=use_multiplicity,
                                                use_voronoi_area=use_voronoi_area,
                                                use_line_graph=use_line_graph,
                                                line_graph_direction=line_graph_direction)

    mb_dataset_cache = MatbenchDataset(dataset_cache)
    
    mb = MatbenchBenchmark(subset=matbench_datasets_subset, autoload=False)
    for idx_task, task in enumerate(mb.tasks):

        # Create/Access cached file for preprocessed crystals
        preprocessed_crystals_file = mb_dataset_cache.get_dataset_file(task.dataset_name, crystal_preprocessor)
        f = h5py.File(preprocessed_crystals_file, 'r')
        preprocessed_crystals = HDFGraphTuple(f)

        # Mapping from unique crystal ids in dataset to index of the cached file
        id_index_mapping = {id_.decode():i for i, id_ in enumerate(preprocessed_crystals.graph_attributes['dataset_id'][:])}
        def get_graphs(inputs):
            idxs = [id_index_mapping[id_] for id_ in inputs.index]
            return preprocessed_crystals[idxs]

        task.load()
        for fold in task.folds:
            intermediate_results_ = results_cache / task.dataset_name / str(fold)
            if intermediate_results_.exists():
                print('Load intermediate results')
                predictions = np.load(str(intermediate_results_ / 'predictions.npy'))
                history_dict = json.load(open(intermediate_results_ / 'history.json', 'r'))
            else:
                # Get training data with target values
                train_inputs, train_outputs = task.get_train_and_val_data(fold)

                train_graphs = get_graphs(train_inputs)
                train_graphs.graph_attributes['label'] = train_outputs.to_numpy()
                train_graphs.graph_attribute_names.append('label')
                x_train, y_train = model_preprocessor.to_ragged_tensors(train_graphs)
                y_train = np.expand_dims(y_train,-1)

                if use_scaler and task.metadata["task_type"] != "classification":
                    scaler = StandardScaler()
                    y_train = scaler.fit_transform(y_train)
                
                if task.metadata["task_type"] == "classification":
                    loss = ks.losses.BinaryCrossentropy(from_logits=True)
                    metrics = [ks.metrics.AUC(from_logits=True)]
                else:
                    loss = ks.losses.MeanAbsoluteError()
                    metrics = [ks.losses.MeanAbsoluteError(), ks.losses.MeanSquaredError()]
                    
                model = get_model(model_cfg.input_block_cfg,
                     [model_cfg.processing_block_cfg]* model_cfg.depth,
                     model_cfg.output_block_cfg,
                     multiplicity=use_multiplicity,
                     voronoi_ridge_area=use_voronoi_area,
                     line_graph=use_line_graph)
                scheduler = get_lr_scheduler(y_train.shape[0], batch_size, epochs,
                                             lr_start=0.0005, lr_stop=1e-5)
                optimizer = ks.optimizers.Adam(learning_rate=scheduler)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                # Train model
                start = datetime.now()
                history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs)
                duration_in_seconds = (datetime.now() - start).total_seconds()

                # Get test data
                test_inputs = task.get_test_data(fold, include_target=False)
                test_graphs = get_graphs(test_inputs)
                test_graphs.graph_attributes['label'] = np.zeros(len(test_inputs)) # dummy values
                test_graphs.graph_attribute_names.append('label')
                x_test, _ = model_preprocessor.to_ragged_tensors(test_graphs)

                # Predict
                predictions = model.predict(x_test)
                
                if use_scaler and task.metadata["task_type"] != "classification":
                    predictions = scaler.inverse_transform(predictions)
                if predictions.shape[-1] == 1:
                    predictions = np.squeeze(predictions, axis=-1)

                intermediate_results_.mkdir(parents=True)
                history_dict = {k: np.array(v).tolist() for k,v in history.history.items()}
                history_dict['training_time'] = duration_in_seconds
                np.save(str(intermediate_results_ / 'predictions.npy'), predictions)
                json.dump(history_dict, open(intermediate_results_ / 'history.json', 'w'))
                model.save_weights(str(intermediate_results_ / 'weights.h5'))

            task.record(fold, predictions, params={
                'input_block': model_cfg.input_block_cfg,
                'processing_blocks': [model_cfg.processing_block_cfg]* model_cfg.depth,
                'output_block': model_cfg.output_block_cfg})
        f.close()
    return mb

if __name__ == '__main__':
    import coGN_config
    matbench_datasets_subset = ["matbench_mp_e_form", "matbench_mp_gap", "matbench_mp_is_metal", "matbench_perovskites",
        "matbench_log_kvrh", "matbench_log_gvrh", "matbench_dielectric", "matbench_phonons",
        "matbench_jdft2d"]
    mb = train_procedure(coGN_config, KNNAsymmetricUnitCell(24),
                         matbench_datasets_subset,
                         use_multiplicity = True,
                         use_voronoi_area = False, use_line_graph = False, line_graph_direction = [1,1],
                         dataset_cache=Path('./dataset_cache'),
                         results_cache=Path('./results'),
                         use_scaler=True,
                         epochs=800,
                         batch_size=64)
    mb.to_file('results.json.gz')

