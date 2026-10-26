import os
import pdb
import json
import torch
import torch_geometric
from torch_geometric.transforms import RadiusGraph, Distance, Spherical, LocalCartesian
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import argparse

import os, json
import pickle, csv
from pathlib import Path

import logging
import sys
from mpi4py import MPI

info = logging.info

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from torch_geometric.data import Data
import random

transform_coordinates = Distance(norm=False, cat=False)

class MatbenchDataset(AbstractBaseDataset):
    """MatbenchDataset datasets class"""
    def __init__(
        self,
        task,
        fold,
        radius = 5.0,
        max_neighbours = 50,
        dist=False,
        test=False,
    ):
        super().__init__()

        self.task = task
        self.fold = fold
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.test = test

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.read_ids()

    def read_ids(self):
        mb = MatbenchBenchmark(autoload=False, subset=[self.task])
        for task in mb.tasks:
            task.load()
            if self.test:
                inputs, outputs = task.get_test_data(self.fold, include_target=True)
            else:
                inputs, outputs = task.get_train_and_val_data(self.fold)
        dataset_len = len(inputs)

        mol_list = []
        if self.dist:
            dataset_len = dataset_len // self.world_size

        for i in range(dataset_len):
            if self.dist:
                ii = (dataset_len // self.world_size)*self.rank + i
            else:
                ii = i
            mol_list.append(self.pmg_to_graph(inputs[ii], outputs[ii]))
        if not self.test:
            random.shuffle(mol_list)
        self.dataset.extend(mol_list)

    def pmg_to_graph(self, molecule, pred):

        atomic_numbers = torch.tensor(molecule.atomic_numbers).unsqueeze(1).to(torch.float64)
        pos = torch.tensor(molecule.cart_coords).to(torch.float64)
        natoms = torch.IntTensor([pos.shape[0]])
        if self.task in ["matbench_mp_is_metal"]:
            pred = torch.tensor(pred, dtype=torch.bool).unsqueeze(0)
        else:
            if self.task in ["matbench_jdft2d"]:
                pred = pred / 1000 # convert to mev/atom to ev/atom
            pred = torch.tensor(pred, dtype=torch.float64).unsqueeze(0)

        charge = 0.0  # neutral
        spin = 1.0  # singlet
        graph_attr = torch.tensor([charge, spin], dtype=torch.float64)

        try:
            x = torch.cat((atomic_numbers, pos), dim=1)

            data_object = Data(
                #dataset_name="matbench",
                natoms=natoms,
                pos=pos,
                atomic_numbers=atomic_numbers,  # Reshaping atomic_numbers to Nx1 tensor
                x=x,
                pred=pred,
                graph_attr=graph_attr,
            )

            data_object.y = data_object.pred

            data_object = self.radius_graph(data_object)

            data_object = transform_coordinates(data_object)

        except AssertionError as e:
            print(f"Assertion error occurred: {e}")

        return data_object

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]



def main():
    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--pretrained_model_ensemble_path", help="directory for ensemble of models", type=str, default="pretrained_model_ensemble"
    )

    parser.add_argument(
        "--finetuning_config", help="path to JSON file with configuration for fine-tunable architecture", type=str,
        default="./finetuning_config.json"
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--modelname", help="model name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    parser.add_argument("--task_name", help="matbench taskname")
    parser.add_argument("--fold", help="dataset fold")

    args = parser.parse_args()

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    modelname = "FineTuning" if args.modelname is None else args.modelname

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_config.json")
    with open(filename, "r") as f:
        ft_config = json.load(f)

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates"]
    node_feature_dims = [1, 3]

    verbosity = ft_config["Verbosity"]["level"]
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    log_name = "matbench_finetuning"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    trainset = MatbenchDataset(
        args.task_name,
        int(args.fold),
    )
    valset = MatbenchDataset(
        args.task_name,
        int(args.fold),
    )
    testset = MatbenchDataset(
        args.task_name,
        int(args.fold),
        test = True,
    )


#    # Use built-in torch_geometric datasets.
#    # Filter function above used to run quick example.
#    # NOTE: data is moved to the device in the pre-transform.
#    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
#    mb = MatbenchBenchmark(autoload=False, subset=[args.task_name])
#    for task in mb.tasks:
#        print(mb.tasks)
#        print(task)
#        task.load()
#        print(task.folds)
#        for fold in task.folds:
#            train_inputs, train_outputs = task.get_train_and_val_data(fold)
#            print(train_inputs[0].cart_coords)
#            print(train_inputs[0].atomic_numbers)
#            print(train_outputs[0])
#            cart_torch = torch.tensor(train_inputs[0].cart_coords)
#            print(cart_torch)

    print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

    print("Before COMM.Barrier()", flush=True)
    comm.Barrier()
    print("After COMM.Barrier()", flush=True)

    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    setnames = ["trainset", "valset", "testset"]

    ## adios
    if args.format == "adios":
        fname = os.path.join(
            os.path.dirname(__file__), "./dataset/%s.bp" % modelname
        )
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

    ## pickle
    elif args.format == "pickle":
        basedir = os.path.join(
            os.path.dirname(__file__), "./dataset", "%s.pickle" % modelname
        )
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
    sys.exit(0)
        




if __name__ == "__main__":

    main()

