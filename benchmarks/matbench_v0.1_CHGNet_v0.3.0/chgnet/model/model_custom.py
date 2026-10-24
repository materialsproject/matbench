from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from pymatgen.core import Structure
from torch import Tensor, nn

from chgnet.graph import CrystalGraph, CrystalGraphConverter
from chgnet.graph.crystalgraph import datatype
from chgnet.model.composition_model import AtomRef
from chgnet.model.encoders import AngleEncoder, AtomEmbedding, BondEncoder
from chgnet.model.functions import MLP, GatedMLP, find_normalization
from chgnet.model.layers import (
    AngleUpdate,
    AtomConv,
    BondConv,
    GraphAttentionReadOut,
    GraphPooling,
)
from chgnet.utils import determine_device

if TYPE_CHECKING:
    from typing_extensions import Self

    from chgnet import PredTask

module_dir = os.path.dirname(os.path.abspath(__file__))


class CHGNetCustomProperty(nn.Module):
    """
    Fine-tuned custom property model 
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | nn.Module = "MPtrj",
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        attn_readout_is_average = True,
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = False,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        graph_converter_algorithm: Literal["legacy", "fast"] = "fast",
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: str | None = "layer",  # noqa: N803
        readout_norm: str | None = "layer",
        version: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize CHGNet.

        Args:
            atom_fea_dim (int): atom feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): bond feature vector embedding dimension.
                Default = 64
            angle_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            composition_model (nn.Module, optional): attach a composition model to
                predict energy or initialize a pretrained linear regression (AtomRef).
                The default 'MPtrj' is the atom reference energy linear regression
                trained on all Materials Project relaxation trajectories
                Default = 'MPtrj'
            num_radial (int): number of radial basis used in bond basis expansion.
                Default = 9
            num_angular (int): number of angular basis used in angle basis expansion.
                Default = 9
            n_conv (int): number of interaction blocks.
                Default = 4
                Note: last interaction block contain only an atom_conv layer
            atom_conv_hidden_dim (List or int): hidden dimensions of
                atom convolution layers.
                Default = 64
            update_bond (bool): whether to use bond_conv_layer in bond graph to
                update bond embeddings
                Default = True.
            bond_conv_hidden_dim (List or int): hidden dimensions of
                bond convolution layers.
                Default = 64
            update_angle (bool): whether to use angle_update_layer to
                update angle embeddings.
                Default = True
            angle_layer_hidden_dim (List or int): hidden dimensions of angle layers.
                Default = 0
            conv_dropout (float): dropout rate in all conv_layers.
                Default = 0
            read_out (str): method for pooling layer, 'ave' for standard
                average pooling, 'attn' for multi-head attention.
                Default = "ave"
            mlp_hidden_dims (int or list): readout multilayer perceptron
                hidden dimensions.
                Default = [64, 64]
            mlp_dropout (float): dropout rate in readout MLP.
                Default = 0.
            is_intensive (bool): whether the energy training label is intensive
                i.e. energy per atom.
                Default = True
            non_linearity ('silu' | 'relu' | 'tanh' | 'gelu'): The name of the
                activation function to use in the gated MLP.
                Default = "silu".
            mlp_first (bool): whether to apply mlp first then pooling.
                if set to True, then CHGNet is essentially calculating energy for each
                atom, them sum them up, this is used for the pretrained model
                Default = True
            atom_graph_cutoff (float): cutoff radius (A) in creating atom_graph,
                this need to be consistent with the value in training dataloader
                Default = 5
            bond_graph_cutoff (float): cutoff radius (A) in creating bond_graph,
                this need to be consistent with value in training dataloader
                Default = 3
            graph_converter_algorithm ('legacy' | 'fast'): algorithm to use
                for converting pymatgen.core.Structure to CrystalGraph.
                'legacy': python implementation of graph creation
                'fast': C implementation of graph creation, this is faster,
                    but will need the cygraph.c file correctly compiled from pip install
                default = 'fast'
            cutoff_coeff (float): cutoff strength used in graph smooth cutoff function.
                the smaller this coeff is, the smoother the basis is
                Default = 5
            learnable_rbf (bool): whether to set the frequencies in rbf and Fourier
                basis functions learnable.
                Default = True
            gMLP_norm (str): normalization layer to use in gate-MLP
                Default = 'layer'
            readout_norm (str): normalization layer to use before readout layer
                Default = 'layer'
            version (str): Pretrained checkpoint version.
            **kwargs: Additional keyword arguments
        """
        # Store model args for reconstruction
        self.model_args = {
            key: val
            for key, val in locals().items()
            if key not in {"self", "__class__", "kwargs"}
        }
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version

        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv

        # Optionally, define composition model
        if isinstance(composition_model, nn.Module):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None

        if self.composition_model is not None:
            # fixed composition_model weights
            for param in self.composition_model.parameters():
                param.requires_grad = False

        # Define Crystal Graph Converter
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            algorithm=graph_converter_algorithm,
            verbose=kwargs.pop("converter_verbose", False),
        )

        # Define embedding layers
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.bond_weights_ag = nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias=False
        )
        self.bond_weights_bg = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias=False
        )

        # Define convolutional layers
        conv_norm = kwargs.pop("conv_norm", None)
        mlp_out_bias = kwargs.pop("mlp_out_bias", False)
        atom_graph_layers = [
            AtomConv(
                atom_fea_dim=atom_fea_dim,
                bond_fea_dim=bond_fea_dim,
                hidden_dim=atom_conv_hidden_dim,
                dropout=conv_dropout,
                activation=non_linearity,
                norm=conv_norm,
                gMLP_norm=gMLP_norm,
                use_mlp_out=True,
                mlp_out_bias=mlp_out_bias,
                resnet=True,
            )
            for _ in range(n_conv)
        ]
        self.atom_conv_layers = nn.ModuleList(atom_graph_layers)

        if update_bond:
            bond_graph_layers = [
                BondConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=bond_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    mlp_out_bias=mlp_out_bias,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = nn.ModuleList(bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]

        if update_angle:
            angle_layers = [
                AngleUpdate(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=angle_layer_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.angle_layers = nn.ModuleList(angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]

        # Define readout layer
        self.site_wise = nn.Linear(atom_fea_dim, 1)
        self.readout_norm = find_normalization(readout_norm, dim=atom_fea_dim)
        self.mlp_first = mlp_first
        if mlp_first: # NOT for us
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}: # THIS IS US
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=attn_readout_is_average
            ) # since weight is softmax processed, it is already averaged, why average again?
            input_dim = atom_fea_dim * num_heads
        else: # THIS IS US
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}: # THIS IS US, prediction head
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else: 
            self.mlp = nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    norm=gMLP_norm,
                    activation=non_linearity,
                ),
                nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )

        version_str = f" v{version}" if version else ""
        print(f"CHGNet{version_str} initialized with {self.n_params:,} parameters")

    @property
    def version(self) -> str | None:
        """Return the version of the loaded checkpoint."""
        return self.model_args.get("version")

    @property
    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        *,
        task: PredTask = "c",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        """Get prediction associated with input graphs
        Args:
            graphs (List): a list of CrystalGraphs
            task (str): the prediction task. One of 'e', 'em', 'ef', 'efs', 'efsm'.
                Default = 'c'
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return the atom features before last
                conv layer.
                Default = False
            return_crystal_feas (bool): whether to return crystal feature.
                Default = False
        Returns:
            model output (dict).
        """
        # # Optionally, make composition model prediction
        # comp_energy = (
        #     0 if self.composition_model is None else self.composition_model(graphs)
        # )

        # Make batched graph
        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.bond_basis_expansion,
            angle_basis_expansion=self.angle_basis_expansion,
            compute_stress=False,
        )

        # Pass to model # AMYAO modified
        custom_property = self._compute(
            batched_graph,
            compute_force=False,
            compute_stress=False,
            compute_magmom=False,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        # custom_property["c"] += comp_energy
       
        return custom_property

    def _compute(
        self,
        g: BatchedGraph,
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_magmom: bool = False,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Args:
            g (BatchedGraph): batched graph
            compute_force (bool): whether to compute force.
                Default = False
            compute_stress (bool): whether to compute stress.
                Default = False
            compute_magmom (bool): whether to compute magmom.
                Default = False
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False

        Returns:
            custom_property
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        prediction = {}
        atoms_per_graph = torch.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph

        # Embed Atoms, Bonds and Angles
        atom_feas = self.atom_embedding(
            g.atomic_numbers - 1
        )  # let H be the first embedding column
        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)
        if len(g.angle_bases) != 0:
            angle_feas = self.angle_embedding(g.angle_bases)

        # Message Passing
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(self.atom_conv_layers[:-1], self.bond_conv_layers, self.angle_layers)
        ):
            # Atom Conv
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )

            # Bond Conv
            if len(g.angle_bases) != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )

                # Angle Update
                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )
            if idx == self.n_conv - 2:
                if return_atom_feas:
                    prediction["atom_fea"] = torch.split(
                        atom_feas, atoms_per_graph.tolist()
                    )
                # Compute site-wise magnetic moments
                if compute_magmom:
                    magmom = torch.abs(self.site_wise(atom_feas))
                    prediction["m"] = list(
                        torch.split(magmom.view(-1), atoms_per_graph.tolist())
                    )

        # Last conv layer
        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=g.batched_atom_graph,
            directed2undirected=g.directed2undirected,
        )
        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)

        # Aggregate nodes and ReadOut, i.e. the prediction head # AMYAO 
        crystal_feas = self.pooling(atom_feas, g.atom_owners)
        custom_property = self.mlp(crystal_feas).view(-1) 
        
        prediction = {}
        prediction["c"] = custom_property

        return prediction

    def get_structure(
        self,
        structure: Structure | Sequence[Structure],
        *,
        batch_size: int = 16,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Predict from pymatgen.core.Structure.

        Args:
            structure (Structure | Sequence[Structure]): structure or a list of
                structures to predict.
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            graphs: list of graphs
        """
        if self.graph_converter is None:
            raise ValueError("graph_converter cannot be None!")

        structures = [structure] if isinstance(structure, Structure) else structure

        graphs = [self.graph_converter(struct) for struct in structures]
        return graphs

    def as_dict(self) -> dict:
        """Return the CHGNet weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}

    def todict(self) -> dict:
        """Needed for ASE JSON serialization when saving CHGNet potential to
        trajectory file (https://github.com/CederGroupHub/chgnet/issues/48).
        """
        return {"model_name": type(self).__name__, "model_args": self.model_args}

    @classmethod
    def from_dict(cls, dct: dict, **kwargs) -> Self:
        """Build a CHGNetCustomProperty from a saved dictionary."""
        args = dct["model_args"]
        args['mlp_first'] = False
        model = cls(**args, **kwargs)
        # model.load_state_dict(dct["state_dict"])
        """ 
        load pre-trained weights
        """
        all_params = ['composition_model.fc.weight', 'atom_embedding.embedding.weight', 'bond_basis_expansion.rbf_expansion_ag.frequencies', 
                    'bond_basis_expansion.rbf_expansion_bg.frequencies', 'bond_embedding.weight',
                    'bond_weights_ag.weight', 'bond_weights_bg.weight', 
                    'angle_basis_expansion.fourier_expansion.frequencies', 
                    'angle_embedding.weight', 
                    'atom_conv_layers.0.twoBody_atom.mlp_core.layers.0.weight', 
                    'atom_conv_layers.0.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.0.twoBody_atom.bn1.weight', 
                    'atom_conv_layers.0.twoBody_atom.bn1.bias', 
                    'atom_conv_layers.0.twoBody_atom.bn2.weight', 
                    'atom_conv_layers.0.twoBody_atom.bn2.bias', 
                    'atom_conv_layers.0.mlp_out.layers.1.weight', 
                    'atom_conv_layers.1.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.1.twoBody_atom.bn1.weight', 'atom_conv_layers.1.twoBody_atom.bn1.bias', 'atom_conv_layers.1.twoBody_atom.bn2.weight', 'atom_conv_layers.1.twoBody_atom.bn2.bias', 'atom_conv_layers.1.mlp_out.layers.1.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.2.twoBody_atom.bn1.weight', 'atom_conv_layers.2.twoBody_atom.bn1.bias', 'atom_conv_layers.2.twoBody_atom.bn2.weight', 'atom_conv_layers.2.twoBody_atom.bn2.bias', 'atom_conv_layers.2.mlp_out.layers.1.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.3.twoBody_atom.bn1.weight', 'atom_conv_layers.3.twoBody_atom.bn1.bias', 'atom_conv_layers.3.twoBody_atom.bn2.weight', 'atom_conv_layers.3.twoBody_atom.bn2.bias', 'atom_conv_layers.3.mlp_out.layers.1.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.0.twoBody_bond.bn1.weight', 'bond_conv_layers.0.twoBody_bond.bn1.bias', 'bond_conv_layers.0.twoBody_bond.bn2.weight', 'bond_conv_layers.0.twoBody_bond.bn2.bias', 'bond_conv_layers.0.mlp_out.layers.1.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.1.twoBody_bond.bn1.weight', 'bond_conv_layers.1.twoBody_bond.bn1.bias', 'bond_conv_layers.1.twoBody_bond.bn2.weight', 'bond_conv_layers.1.twoBody_bond.bn2.bias', 'bond_conv_layers.1.mlp_out.layers.1.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.2.twoBody_bond.bn1.weight', 'bond_conv_layers.2.twoBody_bond.bn1.bias', 'bond_conv_layers.2.twoBody_bond.bn2.weight', 'bond_conv_layers.2.twoBody_bond.bn2.bias', 'bond_conv_layers.2.mlp_out.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.0.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.0.twoBody_bond.bn1.weight', 'angle_layers.0.twoBody_bond.bn1.bias', 'angle_layers.0.twoBody_bond.bn2.weight', 'angle_layers.0.twoBody_bond.bn2.bias', 'angle_layers.1.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.1.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.1.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.1.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.1.twoBody_bond.bn1.weight', 'angle_layers.1.twoBody_bond.bn1.bias', 'angle_layers.1.twoBody_bond.bn2.weight', 'angle_layers.1.twoBody_bond.bn2.bias', 'angle_layers.2.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.2.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.2.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.2.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.2.twoBody_bond.bn1.weight', 'angle_layers.2.twoBody_bond.bn1.bias', 'angle_layers.2.twoBody_bond.bn2.weight', 'angle_layers.2.twoBody_bond.bn2.bias', 
                    'site_wise.weight', 'site_wise.bias', 'readout_norm.weight', 'readout_norm.bias', 
                    'mlp.layers.0.weight', 'mlp.layers.0.bias', 
                    'mlp.layers.2.weight', 'mlp.layers.2.bias', 
                    'mlp.layers.4.weight', 'mlp.layers.4.bias', 
                    'mlp.layers.7.weight', 'mlp.layers.7.bias']
        # See https://discuss.pytorch.org/t/loading-a-specific-layer-from-checkpoint/52725/2
        # on how to load weights for specific layers
        with torch.no_grad():
            for param_name in all_params:
                if 'mlp' in param_name:
                    pass # this is the new prediction head that we need to train
                else:
                    param = dict(model.named_parameters())[param_name]
                    param.copy_(dct['state_dict'][param_name])
        # return 
        return model

    @classmethod
    def from_file(cls, path: str, **kwargs) -> Self:
        """Build a CHGNetCustomProperty from a saved file."""
        state = torch.load(path, map_location=torch.device("cpu"))
        return cls.from_dict(state["model"], **kwargs)

    @classmethod
    def load(
        cls,
        *,
        model_name: str = "0.3.0",
        use_device: str | None = None,
        check_cuda_mem: bool = False,
        verbose: bool = True,
        final_mlp = "MLP",
    ) -> Self:
        """Load pretrained CHGNet model.

        Args:
            model_name (str, optional):
                Default = "0.3.0".
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            verbose (bool): whether to print model device information
                Default = True
        Raises:
            ValueError: On unknown model_name.
        """
        checkpoint_path = {
            "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
            "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        }.get(model_name)

        if checkpoint_path is None:
            raise ValueError(f"Unknown {model_name=}")
        
        mlp_out_bias = (model_name == "0.2.0")
        # mlp_out_bias=True is set for backward compatible behavior but in rare
        # cases causes unphysical jumps in bonding energy. see
        # https://github.com/CederGroupHub/chgnet/issues/79


        model = cls.from_file(
            os.path.join(module_dir, checkpoint_path),
            mlp_out_bias=mlp_out_bias,
            version=model_name,
            final_mlp = final_mlp,
        )

        # Determine the device to use
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)

        # Move the model to the specified device
        model = model.to(device)
        if verbose:
            print(f"CHGNetCustomProperty will run on {device}")
        return model
    
    @classmethod
    def load_w_attn(
        cls,
        *,
        model_name: str = "0.3.0",
        use_device: str | None = None,
        check_cuda_mem: bool = False,
        verbose: bool = True,
        attn_readout_is_average = True,
        final_mlp = "MLP",
    ) -> Self:
        """Load pretrained CHGNet model, with attention read out layer

        Args:
            model_name (str, optional):
                Default = "0.3.0".
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            verbose (bool): whether to print model device information
                Default = True
        Raises:
            ValueError: On unknown model_name.
        """
        checkpoint_path = {
            "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
            "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        }.get(model_name)

        if checkpoint_path is None:
            raise ValueError(f"Unknown {model_name=}")
        
        state = torch.load(os.path.join(module_dir, checkpoint_path), map_location=torch.device("cpu"))
        dct = state['model']
        
        
        args = dct["model_args"]
        args['mlp_first'] = False
        args['read_out']= 'attn'
        args['attn_readout_is_average'] = attn_readout_is_average
        
        mlp_out_bias = (model_name == "0.2.0")
        # mlp_out_bias=True is set for backward compatible behavior but in rare
        # cases causes unphysical jumps in bonding energy. see
        # https://github.com/CederGroupHub/chgnet/issues/79
        kwargs = {"mlp_out_bias": mlp_out_bias, "version": model_name, "final_mlp": final_mlp }
        model = cls(**args, **kwargs)
        """ 
        load pre-trained weights
        """
        all_params = ['composition_model.fc.weight', 'atom_embedding.embedding.weight', 'bond_basis_expansion.rbf_expansion_ag.frequencies', 'bond_basis_expansion.rbf_expansion_bg.frequencies', 'bond_embedding.weight', 'bond_weights_ag.weight', 'bond_weights_bg.weight', 'angle_basis_expansion.fourier_expansion.frequencies', 'angle_embedding.weight', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.0.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.0.twoBody_atom.bn1.weight', 'atom_conv_layers.0.twoBody_atom.bn1.bias', 'atom_conv_layers.0.twoBody_atom.bn2.weight', 'atom_conv_layers.0.twoBody_atom.bn2.bias', 'atom_conv_layers.0.mlp_out.layers.1.weight', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.1.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.1.twoBody_atom.bn1.weight', 'atom_conv_layers.1.twoBody_atom.bn1.bias', 'atom_conv_layers.1.twoBody_atom.bn2.weight', 'atom_conv_layers.1.twoBody_atom.bn2.bias', 'atom_conv_layers.1.mlp_out.layers.1.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.2.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.2.twoBody_atom.bn1.weight', 'atom_conv_layers.2.twoBody_atom.bn1.bias', 'atom_conv_layers.2.twoBody_atom.bn2.weight', 'atom_conv_layers.2.twoBody_atom.bn2.bias', 'atom_conv_layers.2.mlp_out.layers.1.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.0.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.0.bias', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.3.weight', 'atom_conv_layers.3.twoBody_atom.mlp_core.layers.3.bias', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0.weight', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0.bias', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3.weight', 'atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3.bias', 'atom_conv_layers.3.twoBody_atom.bn1.weight', 'atom_conv_layers.3.twoBody_atom.bn1.bias', 'atom_conv_layers.3.twoBody_atom.bn2.weight', 'atom_conv_layers.3.twoBody_atom.bn2.bias', 'atom_conv_layers.3.mlp_out.layers.1.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.0.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.0.twoBody_bond.bn1.weight', 'bond_conv_layers.0.twoBody_bond.bn1.bias', 'bond_conv_layers.0.twoBody_bond.bn2.weight', 'bond_conv_layers.0.twoBody_bond.bn2.bias', 'bond_conv_layers.0.mlp_out.layers.1.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.1.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.1.twoBody_bond.bn1.weight', 'bond_conv_layers.1.twoBody_bond.bn1.bias', 'bond_conv_layers.1.twoBody_bond.bn2.weight', 'bond_conv_layers.1.twoBody_bond.bn2.bias', 'bond_conv_layers.1.mlp_out.layers.1.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.0.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.0.bias', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.3.weight', 'bond_conv_layers.2.twoBody_bond.mlp_core.layers.3.bias', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0.weight', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0.bias', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3.weight', 'bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3.bias', 'bond_conv_layers.2.twoBody_bond.bn1.weight', 'bond_conv_layers.2.twoBody_bond.bn1.bias', 'bond_conv_layers.2.twoBody_bond.bn2.weight', 'bond_conv_layers.2.twoBody_bond.bn2.bias', 'bond_conv_layers.2.mlp_out.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.0.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.0.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.0.twoBody_bond.bn1.weight', 'angle_layers.0.twoBody_bond.bn1.bias', 'angle_layers.0.twoBody_bond.bn2.weight', 'angle_layers.0.twoBody_bond.bn2.bias', 'angle_layers.1.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.1.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.1.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.1.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.1.twoBody_bond.bn1.weight', 'angle_layers.1.twoBody_bond.bn1.bias', 'angle_layers.1.twoBody_bond.bn2.weight', 'angle_layers.1.twoBody_bond.bn2.bias', 'angle_layers.2.twoBody_bond.mlp_core.layers.1.weight', 'angle_layers.2.twoBody_bond.mlp_core.layers.1.bias', 'angle_layers.2.twoBody_bond.mlp_gate.layers.1.weight', 'angle_layers.2.twoBody_bond.mlp_gate.layers.1.bias', 'angle_layers.2.twoBody_bond.bn1.weight', 'angle_layers.2.twoBody_bond.bn1.bias', 'angle_layers.2.twoBody_bond.bn2.weight', 'angle_layers.2.twoBody_bond.bn2.bias', 
                      'site_wise.weight', 'site_wise.bias', 'readout_norm.weight', 'readout_norm.bias', 
                      'pooling.key.layers.0.weight', 'pooling.key.layers.0.bias', 
                      'pooling.key.layers.3.weight', 'pooling.key.layers.3.bias', 
                      'mlp.layers.0.weight', 'mlp.layers.0.bias', 
                      'mlp.layers.2.weight', 'mlp.layers.2.bias', 
                      'mlp.layers.4.weight', 'mlp.layers.4.bias', 
                      'mlp.layers.7.weight', 'mlp.layers.7.bias']
        # See https://discuss.pytorch.org/t/loading-a-specific-layer-from-checkpoint/52725/2
        # on how to load weights for specific layers
        with torch.no_grad():
            for param_name in all_params:
                if 'mlp' in param_name or "pooling" in param_name:
                    pass # this is the new prediction head that we need to train
                else:
                    param = dict(model.named_parameters())[param_name]
                    param.copy_(dct['state_dict'][param_name])
        

        # Determine the device to use
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)

        # Move the model to the specified device
        model = model.to(device)
        if verbose:
            print(f"CHGNetCustomProperty will run on {device}")
        return model


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_ag, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_bg, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 3]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (list[Tensor]): cartesian coordinates of the atoms
            from structures
            [[num_atoms_1, 3], [num_atoms_2, 3], ...]
        strains (list[Tensor]): a list of strains that's initialized to be zeros
            [[3, 3], [3, 3], ...]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
    """

    atomic_numbers: Tensor
    bond_bases_ag: Tensor
    bond_bases_bg: Tensor
    angle_bases: Tensor
    batched_atom_graph: Tensor
    batched_bond_graph: Tensor
    atom_owners: Tensor
    directed2undirected: Tensor
    atom_positions: Sequence[Tensor]
    strains: Sequence[Tensor]
    volumes: Sequence[Tensor] | Tensor

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: nn.Module,
        angle_basis_expansion: nn.Module,
        *,
        compute_stress: bool = False,
    ) -> Self:
        """Featurize and assemble a list of graphs.

        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
            bond_basis_expansion (nn.Module): bond basis expansion layer in CHGNet
            angle_basis_expansion (nn.Module): angle basis expansion layer in CHGNet
            compute_stress (bool): whether to compute stress. Default = False

        Returns:
            BatchedGraph: assembled graphs ready for batched CHGNet forward pass
        """
        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases = [], [], []
        batched_atom_graph, batched_bond_graph = [], []
        directed2undirected = []
        atom_owners = []
        atom_offset_idx = n_undirected = 0

        for graph_idx, graph in enumerate(graphs):
            # Atoms
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)

            # Lattice
            if compute_stress:
                strain = graph.lattice.new_zeros([3, 3], requires_grad=True)
                lattice = graph.lattice @ (
                    torch.eye(3, dtype=datatype).to(strain.device) + strain
                )
            else:
                strain = None
                lattice = graph.lattice
            volumes.append(
                torch.dot(lattice[0], torch.linalg.cross(lattice[1], lattice[2]))
            )
            strains.append(strain)

            # Bonds
            atom_cart_coords = graph.atom_frac_coord @ lattice
            if graph.atom_graph.dim() == 1:
                # This is to avoid structure with all atoms isolated
                graph.atom_graph = graph.atom_graph.reshape(0, 2)
            bond_basis_ag, bond_basis_bg, bond_vectors = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)

            # Indexes
            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(graph.directed2undirected + n_undirected)

            # Angles
            # Here we use directed edges to calculate angles, and
            # keep only the undirected graph index in the bond_graph,
            # So the number of columns in bond_graph reduce from 5 to 3
            if len(graph.bond_graph) != 0:
                bond_vecs_i = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 2]
                )
                bond_vecs_j = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 4]
                )
                angle_basis = angle_basis_expansion(bond_vecs_i, bond_vecs_j)
                angle_bases.append(angle_basis)

                bond_graph = graph.bond_graph.new_zeros([graph.bond_graph.shape[0], 3])
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)

            atom_owners.append(torch.ones(n_atom, requires_grad=False) * graph_idx)
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)

        # Make Torch Tensors
        atomic_numbers = torch.cat(atomic_numbers, dim=0)
        bond_bases_ag = torch.cat(bond_bases_ag, dim=0)
        bond_bases_bg = torch.cat(bond_bases_bg, dim=0)
        angle_bases = (
            torch.cat(angle_bases, dim=0) if len(angle_bases) != 0 else torch.tensor([])
        )
        batched_atom_graph = torch.cat(batched_atom_graph, dim=0)
        if batched_bond_graph != []:
            batched_bond_graph = torch.cat(batched_bond_graph, dim=0)
        else:  # when bond graph is empty or disabled
            batched_bond_graph = torch.tensor([])
        atom_owners = (
            torch.cat(atom_owners, dim=0).type(torch.int32).to(atomic_numbers.device)
        )
        directed2undirected = torch.cat(directed2undirected, dim=0)
        volumes = torch.tensor(volumes, dtype=datatype, device=atomic_numbers.device)

        return cls(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )
