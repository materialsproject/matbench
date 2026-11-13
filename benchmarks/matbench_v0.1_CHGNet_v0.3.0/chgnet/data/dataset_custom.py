from __future__ import annotations

import functools
import os
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from chgnet import utils
from chgnet.graph import CrystalGraph, CrystalGraphConverter

from matbench.bench import MatbenchBenchmark

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from chgnet import TrainTask

warnings.filterwarnings("ignore")
datatype = torch.float32

random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class StructureDataCustomProperty(Dataset):
    """A simple torch Dataset of structures."""

    def __init__(
        self,
        structures: list[Structure],
        custom_property: list[float],
        structure_ids: list | None = None,
        graph_converter: CrystalGraphConverter | None = None,
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            structures (list[dict]): pymatgen Structure objects.
            custom_property (list[float]): [data_size, 1]
            structure_ids (list, optional): a list of ids to track the structures
                Default = None
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True

        Raises:
            RuntimeError: if the length of structures and labels (energies, forces,
                stresses, magmoms) are not equal.
        """
        for idx, struct in enumerate(structures):
            if not isinstance(struct, Structure):
                raise TypeError(f"{idx} is not a pymatgen Structure object: {struct}")
        for name in "custom_property".split():
            labels = locals()[name]
            if labels is not None and len(labels) != len(structures):
                raise RuntimeError(
                    f"Inconsistent number of structures and labels: "
                    f"{len(structures)=}, len({name})={len(labels)}"
                )
        self.structures = structures
        self.custom_property = custom_property
        self.structure_ids = structure_ids
        self.keys = np.arange(len(structures))
        if shuffle:
            random.shuffle(self.keys)
        print(f"{type(self).__name__} imported {len(structures):,} structures")
        self.graph_converter = graph_converter or CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3
        )
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    @classmethod
    def from_file(
        cls,
        file_root: str,
        *,
        graph_converter: CrystalGraphConverter | None = None,
        shuffle: bool = True,
    ) -> Self:
        """Parse VASP output files into structures and labels and feed into the dataset.

        Args:
            file_root (str): the directory of the VASP calculation outputs
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        
        import pandas as pd
        working_dir = os.getcwd()
        os.chdir(file_root)
        structures_list = []
        custom_property_list = []
        structure_ids_list = []
        # read id_prop.csv 
        df = pd.read_csv("id_prop.csv", header=None)
        structure_ids_list = df[0].to_list()
        custom_property_list = df[1].to_list()
        for i in range(0, len(structure_ids_list)):
            structure_now = Structure.from_file(str(structure_ids_list[i]))
            structures_list.append(structure_now)
        os.chdir(working_dir)
        return cls(
            structures=structures_list,
            custom_property=custom_property_list,
            structure_ids=structure_ids_list,
            graph_converter=graph_converter,
            shuffle=shuffle,
        )
    
    @classmethod
    def from_matbench(
        cls,
        fold: float,
        dataset_name: str,
        is_train: bool,
        include_target: bool = True,
        *,
        graph_converter: CrystalGraphConverter | None = None,
        shuffle: bool = True,
    ) -> Self:
        mb = MatbenchBenchmark(autoload=False,subset=[dataset_name])
        for task in mb.tasks:
            task.load()
            if is_train:   
                df = task.get_train_and_val_data(fold, as_type="df")
            else:
                df = task.get_test_data(fold, as_type="df", include_target=include_target)
        structures_list = df[df.keys()[0]].tolist()
        custom_property_list = df[df.keys()[1]].tolist()
        return cls(
            structures=structures_list,
            custom_property=custom_property_list,
            graph_converter=graph_converter,
            shuffle=shuffle,
        )     

    def __len__(self) -> int:
        """Get the number of structures in this dataset."""
        return len(self.keys)

    @functools.cache  # Cache loaded structures
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict]:
        """Get one graph for a structure in this dataset.

        Args:
            idx (int): Index of the structure

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): list of targets. i.e. energy, force, stress
        """
        if idx not in self.failed_idx:
            graph_id = self.keys[idx]
            try:
                struct = self.structures[graph_id]
                if self.structure_ids is not None:
                    mp_id = self.structure_ids[graph_id]
                else:
                    mp_id = graph_id
                crystal_graph = self.graph_converter(
                    struct, graph_id=graph_id, mp_id=mp_id
                )
                
                targets = {
                    "c": torch.tensor(self.custom_property[graph_id], dtype=datatype),
                }

                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another randomly selected
            # structure
            except Exception:
                struct = self.structures[graph_id]
                self.failed_graph_id[graph_id] = struct.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


def collate_graphs(batch_data: list) -> tuple[list[CrystalGraph], dict[str, Tensor]]:
    """Collate of list of (graph, target) into batch data.

    Args:
        batch_data (list): list of (graph, target(dict))

    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            custom_property (Tensor): custom_property of the structures [batch_size]
    """
    graphs = [graph for graph, _ in batch_data]
    all_targets = {key: [] for key in batch_data[0][1]}
    all_targets["c"] = torch.tensor(
        [targets["c"] for _, targets in batch_data], dtype=datatype
    )
    return graphs, all_targets


def get_train_val_test_loader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Randomly partition a dataset into train, val, test loaders.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        train_ratio (float): The ratio of the dataset to use for training
            Default = 0.8
        val_ratio (float): The ratio of the dataset to use for validation
            Default: 0.1
        return_test (bool): Whether to return a test data loader
            Default = True
        num_workers (int): The number of worker processes for loading the data
            see torch Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        train_loader, val_loader and optionally test_loader
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices[0:train_size]),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(
            indices=indices[train_size : train_size + val_size]
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=SubsetRandomSampler(indices=indices[train_size + val_size :]),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def get_loader(
    dataset, *, batch_size: int = 64, num_workers: int = 0, pin_memory: bool = True
) -> DataLoader:
    """Get a dataloader from a dataset.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        num_workers (int): The number of worker processes for loading the data
            see torch Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        data_loader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
