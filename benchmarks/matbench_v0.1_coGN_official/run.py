from typing import Union, Iterable
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as ks

from matbench.bench import MatbenchBenchmark
from pymatgen.core.structure import Structure
from networkx import MultiDiGraph
from graphlist import GraphList, HDFGraphList
from kgcnn.literature.coGN import make_model, model_default, model_default_nested
from kgcnn.crystal.preprocessor import KNNUnitCell, KNNAsymmetricUnitCell, CrystalPreprocessor, VoronoiAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices

class MatbenchDataset:
    """MatbenchDataset class for graph representations of the Matbench datasets.
    This class computes graph representations for crystal structures in the matbench datasets
    and caches them on the disk (in a specified directory as HDF files) for future use.
    This way crystal preprocessing is done only once for each dataset-preprocessor-combination.
    Upon instantiation of the class the user has to specify the directory where cached files are stored.
    Afterwards the user can call `get_dataset_file` to get the file to the preprocessed crystals
    for a dataset-preprocessor-combination..
    In case this preprocessor has been used for the dataset-preprocessor-combination it simply returns the file path,
    otherwise it computes the graph, stores it in a new file and returns this file path.
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """Instantiates a MatbenchDataset object for a given cache directory.
        Args:
            cache_dir (Union[str, Path]): Path to the directory where preprocessed crystal graphs should be stored.
        """
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        if not cache_dir.is_dir():
            cache_dir.mkdir(exist_ok=True)
        self.matbench = MatbenchBenchmark(autoload=False)
        self.dataset_names = [task.dataset_name for task in self.matbench.tasks]

    @staticmethod
    def crystal_iterator(crystal_series: pd.Series):
        """Iterator over pymatgen structures with meta information, which is needed to create the crystal graphs.
        Args:
            crystal_series (pd.Series): Pandas Series of the crystals (as provided by matbench python package).
        Yields:
             pymatgen structures with meta information
        """
        for id_, x in zip(crystal_series.index, crystal_series):
            setattr(x, "dataset_id", id_.encode())
            yield x

    def dataset_exists(
        self, dataset_name: str, preprocessor: CrystalPreprocessor
    ) -> bool:
        """Checks whether a given dataset-preprocessor-combination exists as precomputed file.
        Args:
            dataset_name (str): Matbench dataset name.
            preprocessor (CrystalPreprocessor): Crystal preprocessor for graph creation.
        Returns:
            bool: Whether a given dataset-preprocessor-combination exists
        """
        assert dataset_name in self.dataset_names
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name
        if not dataset_cache_dir.is_dir():
            return False
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        return hdf_file.is_file()

    def get_dataset_file(
        self,
        dataset_name: str,
        preprocessor: CrystalPreprocessor,
        processes=8,
        batch_size=500,
    ):
        """Returns the file path for a dataset-preprocessor-combination.
        If the dataset-preprocessor-combination was used before it returns the cached version.
        If not it first applies preprocessing to all crystals to create the graph dataset.
        Args:
            dataset_name (str): Matbench dataset name.
            preprocessor (CrystalPreprocessor): Crystal preprocessor for graph creation.
            processes (int, optional): How many parallel processes to use for graph dataset creation. Defaults to 5.
            batch_size (int, optional): How big batches are for graph dataset creation.
                On each batch several processes are started in parallel,
                after all parallel processes are done the results are written to the file,
                before the next batch is processed.
                Defaults to 1000.
        Returns:
            Path: The path to the HDF file which contains preprocessed crystals.
        """
        assert dataset_name in self.dataset_names
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name
        if not dataset_cache_dir.is_dir():
            dataset_cache_dir.mkdir(exist_ok=True)
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        if hdf_file.is_file():
            return hdf_file

        full_dataset = self._get_full_dataset(dataset_name)
        hdf_file_ = create_graph_dataset(
            self.crystal_iterator(full_dataset),
            preprocessor,
            hdf_file,
            additional_graph_attributes=["dataset_id"],
            processes=processes,
            batch_size=batch_size,
        )
        with open(str(hdf_file) + ".json", "w") as meta_data_file:
            json.dump(preprocessor.get_config(), meta_data_file)
        return hdf_file_

    def _get_full_dataset(self, dataset_name: str):
        assert dataset_name in self.dataset_names
        task = getattr(self.matbench, dataset_name)
        task.load()
        train_input = task.get_train_and_val_data(0)[0]
        test_input = task.get_test_data(0)
        all_crystals = pd.concat([train_input, test_input]).sort_index()
        return all_crystals


class PreprocessorWrapper:
    """Callable that modifies the behaviour of CrystalProcessors to include extra global graph attributes.
    Returns:
        MultiDiGraph: Crystal graph with extra global graph attributes.
    """

    def __init__(self, preprocessor, additional_graph_attributes=[]):
        self.preprocessor = preprocessor
        self.additional_graph_attributes = additional_graph_attributes

    def __call__(self, crystal: Union[MultiDiGraph, Structure]):
        graph = self.preprocessor(crystal)
        for attribute in self.additional_graph_attributes:
            setattr(graph, attribute, getattr(crystal, attribute))
        return graph


def batcher(iterable, batch_size):
    """Creates batches for iterables"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def create_graph_dataset(
    crystals: Iterable[Structure],
    preprocessor: CrystalPreprocessor,
    out_file: Path,
    additional_graph_attributes=[],
    processes=1,
    batch_size=10) -> Path:
    """Creates a HDF file containing crystal graphs.
    Args:
        crystals (Union[Iterable[MetaDataWrapper], Iterable[Structure]]): Iterable of pymatgen structures.
        preprocessor (CrystalPreprocessor): The preprocessor to use for crystal graph creation.
        out_file (Path): The file to write the crystal graphs to.
        additional_graph_attributes (list, optional): Strings of additional graph attributes to include in the HDF file.
            Defaults to [].
        processes (int, optional): How many processes to use for preprocessing. Defaults to None.
        batch_size (int, optional): The batch size for parallel preprocessing. Defaults to 1000.
    """
    worker = PreprocessorWrapper(preprocessor, additional_graph_attributes=additional_graph_attributes)
    with h5py.File(str(out_file), "w") as f:
        with Pool(processes) as p:
            for batch in batcher(crystals, batch_size):
                graphs = p.map(worker, batch)
                graphlist = GraphList.from_nx_graphs(graphs,
                    node_attribute_names=preprocessor.node_attributes,
                    edge_attribute_names=preprocessor.edge_attributes,
                    graph_attribute_names=preprocessor.graph_attributes
                    + additional_graph_attributes)
                # Append graphlist to HDF5 file
                HDFGraphList.from_graphlist(f, graphlist)
            f.attrs["preprocessor_config"] = json.dumps(
                preprocessor.get_config(), indent=2
            )
    return out_file



def get_lr_scheduler(dataset_size, batch_size, epochs, lr_start=0.0005, lr_stop=1e-5):
    steps_per_epoch = dataset_size / batch_size
    num_steps = epochs * steps_per_epoch
    scheduler = ks.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr_start,
                                                        decay_steps=num_steps,
                                                        end_learning_rate=lr_stop)
    return scheduler

def get_input_tensors(inputs, graphlist):
    """Returns input tensors 
    
    Args:
        inputs (list): Input layers for the model (e.g. `model.inputs`).
        graphlist (GraphList): Data to build tensors from in GraphList form.
    Returns:
        dict: Dictionary of input tensors with {tensorname: tensordata} mapping.
    """
    input_names = [input.name for input in inputs]
    input_tensors = {}
    for input_name in graphlist.edge_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                    graphlist.edge_attributes[input_name],
                    graphlist.num_edges)
    for input_name in graphlist.node_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                    graphlist.node_attributes[input_name],
                    graphlist.num_nodes)
    for input_name in graphlist.graph_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.convert_to_tensor(graphlist.graph_attributes[input_name])
    input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
                    graphlist.edge_indices[:][:, [1, 0]],
                    graphlist.num_edges)
    if 'line_graph_edge_indices' in input_names:
        graphs_line_graph_edge_indices = []
        for g in graphlist:
            line_graph_edge_indices = get_angle_indices(g.edge_indices, edge_pairing='kj')[2].reshape(-1,2) # \measuredangle e_ij e_kj
            graphs_line_graph_edge_indices.append(line_graph_edge_indices) 
        line_graph_edge_indices = tf.RaggedTensor.from_row_lengths(
                np.concatenate(graphs_line_graph_edge_indices),
                [len(l) for l in graphs_line_graph_edge_indices])
        input_tensors['line_graph_edge_indices'] = line_graph_edge_indices

    return input_tensors

def get_id_index_mapping(graphlist):
    index_mapping = {id_.decode():i for i, id_ in 
            enumerate(graphlist.graph_attributes['dataset_id'][:])}
    return index_mapping

def get_graphs(id_index_mapping, graphlist, inputs):
    idxs = [id_index_mapping[id_] for id_ in inputs.index]
    return graphlist[idxs]


def train_procedure(model_cfg, crystal_preprocessor, matbench_datasets_subset, 
                    dataset_cache=Path('./dataset_cache'), results_cache=Path('./results'),
                    use_scaler=True, epochs=800, batch_size=64):

    # MatBench training procedure

    # Directory where to store cached preporcessed crystal graphs (for MatbenchDataset class)
    dataset_cache = Path(dataset_cache)
    mb_dataset_cache = MatbenchDataset(dataset_cache)
    # Results are stored in files in a directory.
    # That way training is only done once per split and dataset, even if the script is interrupted/restarted.
    results_cache = Path(results_cache)
    
    # MatbenchBenchmark for loading unprocessed crystals and labels.
    mb = MatbenchBenchmark(subset=matbench_datasets_subset, autoload=False)

    for task in mb.tasks:
        # Returns file path to preprocessed crystals.
        # If crystals aren't preprocessed yet with the given preprocessor, the file is creates first.
        # Preprocessing a crystal may take a while.
        preprocessed_crystals_file = mb_dataset_cache.get_dataset_file(task.dataset_name, crystal_preprocessor)

        # Load preprocessed cyrstals
        with h5py.File(preprocessed_crystals_file, 'r') as f:
            
            # Load as GraphList data structure
            preprocessed_crystals = HDFGraphList(f)

            # Mapping from unique crystal ids in dataset to index of the cached file
            # This is a helper construct to reconstruct original matbench dataset splits with preprocessed crystals. 
            id_index_mapping = get_id_index_mapping(preprocessed_crystals)

            task.load()
            for fold in task.folds:
                intermediate_results_ = results_cache / task.dataset_name / str(fold)

                if intermediate_results_.exists():
                    # If there are already intermediate results stored, skip the training.
                    print('Load intermediate results')
                    predictions = np.load(str(intermediate_results_ / 'predictions.npy'))
                    history_dict = json.load(open(intermediate_results_ / 'history.json', 'r'))
                else:
                    # Create coGN model with KGCNN
                    model = make_model(**model_cfg)

                    # Get training data with target values
                    train_inputs, train_outputs = task.get_train_and_val_data(fold)

                    train_graphs = get_graphs(id_index_mapping, preprocessed_crystals, train_inputs)
                    # Get Tensor Input
                    x_train = get_input_tensors(model.inputs, train_graphs)
                    y_train = train_outputs.to_numpy().reshape(-1,1)

                    if use_scaler and task.metadata["task_type"] != "classification":
                        scaler = StandardScaler()
                        y_train = scaler.fit_transform(y_train)
                    
                    if task.metadata["task_type"] == "classification":
                        loss = ks.losses.BinaryCrossentropy(from_logits=True)
                        metrics = [ks.metrics.AUC(from_logits=True)]
                    else:
                        loss = ks.losses.MeanAbsoluteError()
                        metrics = [ks.losses.MeanAbsoluteError(), ks.losses.MeanSquaredError()]
                        
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
                    test_graphs = get_graphs(id_index_mapping, preprocessed_crystals, test_inputs)
                    x_test = get_input_tensors(model.inputs, test_graphs)

                    # Predict
                    predictions = model.predict(x_test)
                    
                    if task.metadata["task_type"] != "classification":
                        if use_scaler:
                            predictions = scaler.inverse_transform(predictions)
                    else:
                        def sigmoid(x):
                            return 1 / (1 + np.exp(-x))
                        predictions = sigmoid(predictions)

                    if predictions.shape[-1] == 1:
                        predictions = np.squeeze(predictions, axis=-1)

                    intermediate_results_.mkdir(parents=True)
                    history_dict = {k: np.array(v).tolist() for k,v in history.history.items()}
                    history_dict['training_time'] = duration_in_seconds
                    np.save(str(intermediate_results_ / 'predictions.npy'), predictions)
                    json.dump(history_dict, open(intermediate_results_ / 'history.json', 'w'))
                    model.save_weights(str(intermediate_results_ / 'weights.h5'))

                task.record(fold, predictions, params=model_cfg)
    return mb

if __name__ == '__main__':
    if len(sys.argv) > 1:
        matbench_datasets_subset = sys.argv[1:]
    else:
        matbench_datasets_subset = ["matbench_mp_e_form", "matbench_mp_gap", "matbench_mp_is_metal", "matbench_perovskites",
            "matbench_log_kvrh", "matbench_log_gvrh", "matbench_dielectric", "matbench_phonons",
            "matbench_jdft2d"]

    model_name = 'coGN'

    if model_name == 'coGN':
        mb = train_procedure(model_default, KNNAsymmetricUnitCell(24),
                             matbench_datasets_subset,
                             dataset_cache=Path('./dataset_cache'),
                             results_cache=Path('./results_coGN'),
                             use_scaler=True,
                             epochs=800,
                             batch_size=64)
        mb.to_file('results.json.gz')
    elif model_name == 'coNGN':
        mb = train_procedure(model_default_nested, VoronoiAsymmetricUnitCell(1e-6),
                             matbench_datasets_subset,
                             dataset_cache=Path('./dataset_cache'),
                             results_cache=Path('./results_coNGN'),
                             use_scaler=True,
                             epochs=600,
                             batch_size=64)
        mb.to_file('results_coNGN.json.gz')

