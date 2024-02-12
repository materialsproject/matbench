class MPNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def init(self, fc_features):
        super(MPNNConv, self).init(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """

        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))




############
# 03/08/2023
class MatformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                   bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # simpler version
        # self.lin_msg_update = nn.Linear(out_channels * 3, out_channels)
        # self.layer_norm = nn.LayerNorm(out_channels)
        # self.msg_layer = nn.Sequential(nn.Linear(out_channels, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out)) # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,self.out_channels)

        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3) 
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))) 
        out = self.msg_layer(out)

        # version two, simpler
        # query_i = query_i
        # key_j = key_j
        # alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        # self._alpha = alpha
        # out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        # out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, self.out_channels))) 
        # out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MatformerConv_edge(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                   bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, edge: Union[Tensor, PairTensor], edge_nei_len: OptTensor = None, edge_nei_angle: OptTensor = None):
        # preprocess for edge of shape [num_edges, hidden_dim]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)

        # preprocess for edge_neighbor of shape [num_edges, 3, hidden_dim]
        query_y = self.lin_query(edge_nei_len).view(-1, 3, H, C)
        key_y = self.lin_key(edge_nei_len).view(-1, 3, H, C)
        value_y = self.lin_value(edge_nei_len).view(-1, 3, H, C)

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        query = torch.cat((query_x, query_x, query_x), dim=-1)
        key = torch.cat((key_x, key_y, edge_xy), dim=-1)
        alpha = (query * key) / math.sqrt(self.out_channels * 3)
        out = torch.cat((value_x, value_y, edge_xy), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha))
        out = self.msg_layer(out)

        if self.concat:
            out = out.view(-1, 3, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)
        
        if self.concat:
            out = self.lin_concate(out)
        
        # aggregate the msg
        out = out.sum(dim=1)

        out = F.silu(self.bn(out))

        if self.root_weight:
            x_r = self.lin_skip(edge[1])
            out += x_r

        return out


#####################
# 03/07/2023
#####################


# class MatformerConv_edge(MessagePassing):
#     _alpha: OptTensor

#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout: float = 0.0,
#         edge_dim: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super(MatformerConv_edge, self).__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.beta = beta and root_weight
#         self.root_weight = root_weight
#         self.concat = concat
#         self.dropout = dropout
#         self.edge_dim = edge_dim
#         self._alpha = None

#         if isinstance(in_channels, int):
#             in_channels = (in_channels, in_channels)

#         self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
#         self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
#         self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
#         if edge_dim is not None:
#             self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
#         else:
#             self.lin_edge = self.register_parameter('lin_edge', None)

#         if concat:
#             self.lin_skip = nn.Linear(in_channels[1], out_channels,
#                                    bias=bias)
#             self.lin_concate = nn.Linear(heads * out_channels, out_channels)
#             if self.beta:
#                 self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
#             else:
#                 self.lin_beta = self.register_parameter('lin_beta', None)
#         else:
#             self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
#             if self.beta:
#                 self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
#             else:
#                 self.lin_beta = self.register_parameter('lin_beta', None)
#         self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
#         self.layer_norm = nn.LayerNorm(out_channels * 3)
#         self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
#         # simpler version
#         # self.lin_msg_update = nn.Linear(out_channels * 3, out_channels)
#         # self.layer_norm = nn.LayerNorm(out_channels)
#         # self.msg_layer = nn.Sequential(nn.Linear(out_channels, out_channels), nn.LayerNorm(out_channels))
#         # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
#         self.bn = nn.BatchNorm1d(out_channels)
#         # self.bn = nn.BatchNorm1d(out_channels * heads)
#         self.sigmoid = nn.Sigmoid()
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_key.reset_parameters()
#         self.lin_query.reset_parameters()
#         self.lin_value.reset_parameters()
#         if self.concat:
#             self.lin_concate.reset_parameters()
#         if self.edge_dim:
#             self.lin_edge.reset_parameters()
#         self.lin_skip.reset_parameters()
#         if self.beta:
#             self.lin_beta.reset_parameters()

#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):

#         H, C = self.heads, self.out_channels
#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)
        
#         query = self.lin_query(x[1]).view(-1, H, C)
#         key = self.lin_key(x[0]).view(-1, H, C)
#         value = self.lin_value(x[0]).view(-1, H, C)

#         out = self.propagate(edge_index, query=query, key=key, value=value,
#                              edge_attr=edge_attr, size=None)
#         alpha = self._alpha
#         self._alpha = None

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)
        
#         if self.concat:
#             out = self.lin_concate(out)

#         out = F.silu(self.bn(out)) # after norm and silu

#         if self.root_weight:
#             x_r = self.lin_skip(x[1])
#             if self.lin_beta is not None:
#                 beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
#                 beta = beta.sigmoid()
#                 out = beta * x_r + (1 - beta) * out
#             else:
#                 out += x_r

        
#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out

#     def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
#                 edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:

#         if self.lin_edge is not None:
#             assert edge_attr is not None
#             edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,self.out_channels)

#         query_i = torch.cat((query_i, query_i, query_i), dim=-1)
#         key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
#         alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3) 
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         out = torch.cat((value_i, value_j, edge_attr), dim=-1)
#         out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))) 
#         out = self.msg_layer(out)

#         # version two, simpler
#         # query_i = query_i
#         # key_j = key_j
#         # alpha = (query_i * key_j) / math.sqrt(self.out_channels)
#         # self._alpha = alpha
#         # out = torch.cat((value_i, value_j, edge_attr), dim=-1)
#         # out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, self.out_channels))) 
#         # out = self.msg_layer(out)
#         return out

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')



#####################
# 03/21/2023
#####################

class MatformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 64,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{out_channels}x0e'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{out_channels}x0e',
            ]
        self.ns = ns
        self.nv = nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.skip_linear = nn.Linear(in_channels, out_channels)
        self.v1_v2_linear = nn.Linear(ns, out_channels)

        self.sh = '1x0e + 1x1o + 1x2e'
        self.v2_tp = v2_tp = o3.FullyConnectedTensorProduct(f'{ns}x0e + {nv}x1o + {nv}x2e', '1x0e + 1x1o + 1x2e', f'{out_channels}x0e', shared_weights=False)
        self.v2_fc = nn.Sequential(
            nn.Linear(edge_dim * 3, edge_dim),
            nn.Softplus(),
            nn.Linear(edge_dim, v2_tp.weight_numel)
        )
        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )

        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )
        # MACE
        self.softplus = nn.Softplus()
        self.ln_0e = nn.Parameter(torch.ones(1, 3, 1))
        self.ln_1o = nn.Parameter(torch.ones(1, 3, 1))
        self.ln_2e = nn.Parameter(torch.ones(1, 3, 1))
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor], 
                    lat_len: OptTensor = None):
        edge_vec = data.edge_attr
        n_ = node_feature.shape[0]
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        lat = o3.spherical_harmonics(self.sh, data.atom_lat.view(n_ * 3, 3), normalize=True, normalization='component')
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        irreps = o3.Irreps('1x0e + 1x1o + 1x2e')
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        # the second layer
        lat_l0, lat_l1o, lat_l2e = lat[:, irreps.slices()[0]], lat[:, irreps.slices()[1]], lat[:, irreps.slices()[2]]
        lat_l0 = (lat_l0.view(n_, 3, 1) * self.ln_0e).sum(dim=1)
        lat_l1o = (lat_l1o.view(n_, 3, 3) * self.ln_1o).sum(dim=1)
        lat_l2e = (lat_l2e.view(n_, 3, 5) * self.ln_2e).sum(dim=1)
        lat_vec = torch.cat((lat_l0, lat_l1o, lat_l2e), dim=-1)
        node_v2 = self.v2_tp(node_feature, lat_vec, self.v2_fc(lat_len.view(n_, -1)))
        node_v2 = self.softplus(self.bn(node_v2))
        node_v2 += self.skip_linear(skip_connect)
        
        return node_v2



class MatformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 128,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{out_channels}x0e'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{out_channels}x0e',
            ]
        self.ns = ns
        self.nv = nv
        # for input x mapping
        self.node_linear = nn.Linear(in_channels, ns)
        # for input x mapping to the output
        self.skip_linear = nn.Linear(in_channels, out_channels)
        # for l0 mapping to the output
        self.v1_v2_linear = nn.Linear(ns, out_channels)

        self.sh = '1x0e + 1x1o + 1x2e'
        self.v2_tp = v2_tp = o3.FullyConnectedTensorProduct(f'{ns}x0e + {nv}x1o + {nv}x2e', '1x0e + 1x1o + 1x2e', f'{ns}x0e + {nv}x1o + {nv}x2e', shared_weights=False)
        self.v2_fc = nn.Sequential(
            nn.Linear(ns, ns),
            nn.Softplus(),
            nn.Linear(ns, v2_tp.weight_numel)
        )

        self.v2_tp_2 = v2_tp_2 = o3.FullyConnectedTensorProduct(f'{ns}x0e + {nv}x1o + {nv}x2e', '1x0e + 1x1o + 1x2e', f'{out_channels}x0e', shared_weights=False)
        self.v2_fc_2 = nn.Sequential(
            nn.Linear(ns, ns),
            nn.Softplus(),
            nn.Linear(ns, v2_tp_2.weight_numel)
        )

        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )

        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )

        # MACE
        self.softplus = nn.Softplus()
        self.ln_0e = nn.Parameter(torch.ones(1, ns))
        self.ln_1o = nn.Parameter(torch.ones(1, nv, 1))
        self.ln_2e = nn.Parameter(torch.ones(1, nv, 1))
        self.bn = nn.BatchNorm1d(ns)

        self.ln_0e2 = nn.Parameter(torch.ones(1, ns))
        self.ln_1o2 = nn.Parameter(torch.ones(1, nv, 1))
        self.ln_2e2 = nn.Parameter(torch.ones(1, nv, 1))
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor], 
                    edge_nei_len: OptTensor = None):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        n_ = node_feature.shape[0]
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        ns, nv = self.ns, self.nv
        irreps = o3.Irreps(f'{ns}x0e + {nv}x1o + {nv}x2e')
        # the first layer
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        # the second layer
        node_l0, node_l1o, node_l2e = node_feature[:, irreps.slices()[0]], node_feature[:, irreps.slices()[1]], node_feature[:, irreps.slices()[2]]
        node_l1o, node_l2e = node_l1o.reshape(n_, -1, 3), node_l2e.reshape(n_, -1, 5)
        node_l0 = self.softplus(node_l0)
        node_l0_update = (node_l0 * self.ln_0e).sum(dim=1, keepdim=True)
        node_l1o = (node_l1o * self.ln_1o).sum(dim=1, keepdim=True)
        node_l2e = (node_l2e * self.ln_2e).sum(dim=1, keepdim=True)
        node_feature_vec = torch.cat((node_l0_update, node_l1o.reshape(n_, -1), node_l2e.reshape(n_, -1)), dim=-1)
        node_v2 = self.v2_tp(node_feature, node_feature_vec, self.v2_fc(node_l0))
        node_v2_l0 = node_v2[:, irreps.slices()[0]]
        node_v2_l0 = node_v2_l0 + node_l0
        node_v2_l0 = self.softplus(self.bn(node_v2_l0))
        node_v2[:, irreps.slices()[0]] = node_v2_l0
        # the first layer
        node_feature = self.nlayer_2(node_v2, edge_index, edge_feature, edge_irr)
        # the second layer
        node_l0, node_l1o, node_l2e = node_feature[:, irreps.slices()[0]], node_feature[:, irreps.slices()[1]], node_feature[:, irreps.slices()[2]]
        node_l1o, node_l2e = node_l1o.reshape(n_, -1, 3), node_l2e.reshape(n_, -1, 5)
        node_l0 = self.softplus(node_l0)
        node_l0_update = (node_l0 * self.ln_0e2).sum(dim=1, keepdim=True)
        node_l1o = (node_l1o * self.ln_1o2).sum(dim=1, keepdim=True)
        node_l2e = (node_l2e * self.ln_2e2).sum(dim=1, keepdim=True)
        node_feature_vec = torch.cat((node_l0_update, node_l1o.reshape(n_, -1), node_l2e.reshape(n_, -1)), dim=-1)
        node_v2 = self.v2_tp_2(node_feature, node_feature_vec, self.v2_fc_2(node_l0))
        node_v2 = node_v2 + self.v1_v2_linear(node_l0)
        node_v2 = self.softplus(self.bn2(node_v2))

        node_v2 += self.skip_linear(skip_connect)
        return node_v2


        
        
        # edge_nei_vec = data.edge_nei / data.edge_nei.norm(dim=-1, keepdim=True)
        # edge_irr = torch.cat((self.edge_tp(edge_vec.unsqueeze(1).repeat(1, 3, 1), edge_nei_vec, self.edge_tp_fc(edge_nei_len)).sum(dim=1),
        #                         edge_vec), dim=-1)

# nonlinearity and norm of equi features
        # ns, nv = self.ns, self.nv
        # irreps = o3.Irreps(f'{ns}x0e + {nv}x1o + {nv}x2e')
        # node_l0, node_l1o, node_l1e = node_feature[:, irreps.slices()[0]], node_feature[:, irreps.slices()[1]], node_feature[:, irreps.slices()[2]]
        # node_l1o, node_l1e = node_l1o.reshape(n_, -1, 3), node_l1e.reshape(n_, -1, 3)
        # #   for order = 0 part
        # node_l0 = self.softplus(node_l0)
        # rms_l0 = node_l0.norm(dim=-1, keepdim=True) * (ns ** -0.5)
        # node_l0 = node_l0 / rms_l0.clamp(min = 1e-12) * self.ln_0e
        # #   for order = 1o part
        # l2norm = node_l1o.norm(dim=-1, keepdim=True)
        # rms_l1o = l2norm.norm(dim=-2, keepdim=True) * (nv ** -0.5)
        # node_l1o = node_l1o / rms_l1o.clamp(min = 1e-12) * self.ln_1o
        # #   for order = 1e part
        # l2norme = node_l1e.norm(dim=-1, keepdim=True)
        # rms_l1e = l2norme.norm(dim=-2, keepdim=True) * (nv ** -0.5)
        # node_l1e = node_l1e / rms_l1e.clamp(min = 1e-12) * self.ln_1e
        # node_feature = torch.cat((node_l0, node_l1o.reshape(n_, -1), node_l1e.reshape(n_, -1)), dim=-1)
        # the second layer
        # node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)



class MatformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 128,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        # if use_second_order_repr:
        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{out_channels}x0e'
        ]
        # else:
        #     irrep_seq = [
        #         f'{ns}x0e',
        #         f'{ns}x0e + {nv}x1o',
        #         f'{out_channels}x0e',
        #     ]
        
        self.node_linear = nn.Linear(in_channels, ns)
        # for input x mapping to the output
        self.skip_linear = nn.Linear(in_channels, out_channels)

        self.sh = '1x0e + 1x1o + 1x2e'

        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )

        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False
        )

        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor], 
                    edge_nei_len: OptTensor = None):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        n_ = node_feature.shape[0]
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.bn(node_feature))
        node_feature += self.skip_linear(skip_connect)

        return node_feature




        # ns, nv = self.ns, self.nv
        # irreps = o3.Irreps(f'{ns}x0e + {nv}x1o + {nv}x2e')
        # node_l0, node_l1o, node_l2e = node_feature[:, irreps.slices()[0]], node_feature[:, irreps.slices()[1]], node_feature[:, irreps.slices()[2]]
        # node_l1o, node_l2e = node_l1o.reshape(n_, -1, 3), node_l2e.reshape(n_, -1, 5)
        # #   for order = 0 part
        # node_l0 = self.softplus(node_l0)
        # rms_l0 = node_l0.norm(dim=-1, keepdim=True) * (ns ** -0.5)
        # node_l0 = node_l0 / rms_l0.clamp(min = 1e-12) * self.ln_0e
        # #   for order = 1o part
        # l2norm = node_l1o.norm(dim=-1, keepdim=True)
        # rms_l1o = l2norm.norm(dim=-2, keepdim=True) * (nv ** -0.5)
        # node_l1o = node_l1o / rms_l1o.clamp(min = 1e-12) * self.ln_1o
        # #   for order = 1e part
        # l2norme = node_l2e.norm(dim=-1, keepdim=True)
        # rms_l2e = l2norme.norm(dim=-2, keepdim=True) * (nv ** -0.5)
        # node_l2e = node_l2e / rms_l2e.clamp(min = 1e-12) * self.ln_2e
        # node_feature = torch.cat((node_l0, node_l1o.reshape(n_, -1), node_l2e.reshape(n_, -1)), dim=-1)