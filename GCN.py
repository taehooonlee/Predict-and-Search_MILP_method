import io
import math
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_sparse
from rtdl_num_embeddings import PeriodicEmbeddings
from rtdl_revisiting_models import MLP
from scipy.sparse import coo_array
from torch import Tensor
from torch.nn import LayerNorm, Linear, ModuleList, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter

from evidential_dl import Dirichlet, NormalInvGamma
from helper import ds2type

# from torchtext.vocab import FastText


class GNNPolicy_MILP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(args.negative_slope)
        elif self.args.activation == "relu":
            self.activation = torch.nn.ReLU()
        elif self.args.activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()

        if args.init_x == "emb":
            self.x_emb = Sequential(
                torch.nn.Embedding(args.num_vars, args.hidden_size),
                Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                torch.nn.ReLU(),
                Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                torch.nn.ReLU(),
            )
        elif args.init_x != "rhs":
            self.x_emb = Sequential(
                Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                torch.nn.ReLU(),
                Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                torch.nn.ReLU(),
            )

        mlp_config = {
            "d_out": args.hidden_size,  # For example, a single regression task.
            "n_blocks": 2,
            "d_block": 256,
            "dropout": 0.1,
        }

        self.rhs_emb = Sequential(
            PeriodicEmbeddings(args.n_rhs_features, args.emb_size, lite=False),
            torch.nn.Flatten(),
            MLP(d_in=args.emb_size * args.n_rhs_features, **mlp_config),
        )

        self.milp_convs = ModuleList(
            [
                MILPConv(
                    self.rhs_emb,
                    args.hidden_size,
                    args.hidden_size,
                    weight_l=args.weight_l,
                    weight_r=args.weight_r,
                    org_version=args.org_version,
                    is_lin_c=args.is_lin_c,
                )
                for _ in range(args.n_conv)
            ]
        )

        self.non_lin_activations = ModuleList(
            [
                Sequential(
                    Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                    # LayerNorm(args.hidden_size),
                    self.activation,
                    torch.nn.Dropout(0.1),
                    # Linear(args.hidden_size, args.hidden_size, bias=args.bias),
                )
                for _ in range(args.n_conv)
            ]
        )

        output_module = []
        output_module.append(Linear(args.hidden_size, args.hidden_size))
        output_module.append(torch.nn.ReLU())
        output_module.append(torch.nn.Dropout(0.1))
        output_module.append(Linear(args.hidden_size, args.hidden_size))
        output_module.append(torch.nn.ReLU())
        output_module.append(torch.nn.Dropout(0.1))
        if args.evi_loss:
            if ds2type[args.task_name] == "B":
                output_module.append(Dirichlet(args.hidden_size, 1))
            elif ds2type[args.task_name] == "I":
                output_module.append(NormalInvGamma(args.hidden_size, 1))
        else:
            output_module.append(Linear(args.hidden_size, 1))

        if ds2type[self.args.task_name] == "B":
            output_module.append(torch.nn.Sigmoid())
        self.output_module = Sequential(*output_module)
        # self.linear_rhs = Linear(args.hidden_size, 1, bias=args.bias)

    def forward(self, hyperedge_index, coef, rhs):
        # Embedding layers for decision variables X and RHS of constraints
        if self.args.init_x == "emb":
            # var_idx = hyperedge_index[0] % self.args.num_vars
            num_graphs = (hyperedge_index[0].max() + 1) / self.args.num_vars
            var_idx = list(range(self.args.num_vars)) * int(num_graphs.item())
            x_var = self.x_emb(torch.tensor(var_idx, device=self.args.device))
        elif self.args.init_x == "rhs":
            x_var = None
        else:
            x = torch.empty(hyperedge_index[0].max() + 1, self.args.hidden_size, requires_grad=False, device=self.args.device)
            if self.args.init_x == "uniform":
                torch.nn.init.uniform_(x, a=-1, b=1)
            elif self.args.init_x == "normal":
                torch.nn.init.normal_(x)
            elif self.args.init_x == "uniform_h":
                a = math.sqrt(1.0 / float(self.args.hidden_size))
                torch.nn.init.uniform_(x, a=-a, b=a)
            elif self.args.init_x == "normal_h":
                std = math.sqrt(1.0 / float(self.args.hidden_size))
                torch.nn.init.normal_(x, mean=0.0, std=std)
            elif self.args.init_x == "uniform_h2":
                a = math.sqrt(2.0 / float(self.args.hidden_size))
                torch.nn.init.uniform_(x, a=-a, b=a)
            elif self.args.init_x == "normal_h2":
                std = math.sqrt(2.0 / float(self.args.hidden_size))
                torch.nn.init.normal_(x, mean=0.0, std=std)
            elif self.args.init_x == "xavier_uniform":
                torch.nn.init.xavier_uniform_(x)
            elif self.args.init_x == "xavier_normal":
                torch.nn.init.xavier_normal_(x)
            elif self.args.init_x == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(x, mode="fan_out", nonlinearity="relu")
            elif self.args.init_x == "kaiming_normal":
                torch.nn.init.kaiming_normal_(x, mode="fan_out", nonlinearity="relu")
            x_var = self.x_emb(x)
        # rhs = self.rhs_emb(rhs)

        # MILP colvolution
        if self.args.n_rhs_features == 1:
            rhs = rhs[:, [1]]

        for _, (milp_conv, non_lin_activation) in enumerate(zip(self.milp_convs, self.non_lin_activations)):
            # rhs = rhs if i == 0 else rhs.detach()
            x_var, x_const, emb_rhs = milp_conv(hyperedge_index, coef=coef, rhs=rhs, x_var=x_var)
            x_var = non_lin_activation(x_var)
            if self.args.gnn_norm:
                x_var = F.normalize(x_var, p=2.0, dim=-1)
        self.x_var = x_var
        self.x_const = x_const
        self.emb_rhs = emb_rhs
        # pred_rhs = self.linear_rhs(x_const)
        # A final MLP on the variable features
        # if self.args.evi_loss and ds2type[self.args.task_name] == "I":
        # return self.output_module(x_var).squeeze(-1)
        #     return self.output_module(x_var)
        # return self.output_module(x_var), pred_rhs, scaled_rhs
        # if ds2type[self.args.task_name] == "B":
        #     out = self.output_module(x_var)
        # print(f"out max: {out.max()}")
        # print(f"out min: {out.min()}")
        #     return out.sigmoid()
        # else:
        return self.output_module(x_var)


class MILPConv(MessagePassing):
    def __init__(
        self,
        rhs_emb,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_l: Optional[bool] = True,
        weight_r: Optional[bool] = False,
        org_version: Optional[bool] = False,
        is_lin_c: Optional[bool] = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "sum")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.rhs_emb = rhs_emb
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_l = weight_l
        self.weight_r = weight_r
        self.org_version = org_version
        self.is_lin_c = is_lin_c
        self.layer_norm = LayerNorm(out_channels)
        self.activation = torch.nn.ReLU()
        if self.is_lin_c:
            self.lin_c = Linear(in_channels, out_channels, bias=bias, weight_initializer="glorot")
        if self.weight_l:
            self.lin_l = Linear(in_channels, out_channels, bias=bias, weight_initializer="glorot")
        if self.weight_r:
            self.lin_r = Linear(in_channels, out_channels, bias=bias, weight_initializer="glorot")
        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        if self.is_lin_c:
            self.lin_c.reset_parameters()
        if self.weight_l:
            self.lin_l.reset_parameters()
        if self.weight_r:
            self.lin_r.reset_parameters()

    # @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        hyperedge_index: Tensor,
        *,
        coef: Tensor,
        rhs: Tensor,
        x_var: Optional[Tensor] = None,
        hyperedge_weight: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        # x = self.lin_x(x)
        num_nodes = int(hyperedge_index[0].max()) + 1
        # num_nodes = x.size(0)
        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = hyperedge_index.new_ones(num_edges)  # It can be used to assign the obj function higher importance.
        hyperedge_index, coef = torch_sparse.coalesce(hyperedge_index, coef, num_nodes, num_edges, op="mean")

        # Column(constraints) scaling
        s_c = scatter(coef.abs(), hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        inv_s_c = 1.0 / s_c
        inv_s_c[inv_s_c == float("inf")] = 0
        # print(f"max inv_s_c: {inv_s_c.max()}")

        # # Column(constraints) normalization
        # deg_c = scatter(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        # inv_deg_c = 1.0 / deg_c
        # inv_deg_c[inv_deg_c == float("inf")] = 0
        # # inv_s_c *= inv_deg_c

        # Scaled coefficient and RHS
        diag_idx = torch.tensor([range(len(inv_s_c)), range(len(inv_s_c))], device=inv_s_c.device, dtype=torch.long)
        diag_idx, inv_s_c = torch_sparse.coalesce(diag_idx, inv_s_c, num_edges, num_edges, op="mean")
        scaled_hyperedge_index, scaled_coef = torch_sparse.spspmm(hyperedge_index, coef, diag_idx, inv_s_c, num_nodes, num_edges, num_edges)
        # torch.sparse_coo_tensor(scaled_hyperedge_index, scaled_coef).to_dense()  # for debugging
        rhs[:, [-1]] = inv_s_c.view(-1, 1) * rhs[:, [-1]]
        emb_rhs = self.rhs_emb(rhs)

        # Row(variables) scaling
        s_v = scatter(scaled_coef, scaled_hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        inv_s_v = 1.0 / s_v
        inv_s_v[inv_s_v == float("inf")] = 0
        # print(f"max inv_s_v: {inv_s_v.max()}")

        if x_var is None:
            # x_var = self.propagate(scaled_hyperedge_index.flip([0]), size=(num_edges, num_nodes), x=emb_rhs, coef=scaled_coef, scaling=inv_s_v)
            temp_out_channels = self.out_channels
            self.out_channels = 1
            x_var = self.propagate(scaled_hyperedge_index.flip([0]), size=(num_edges, num_nodes), x=rhs, coef=scaled_coef, scaling=inv_s_v)
            self.out_channels = temp_out_channels
            x_var = self.rhs_emb(x_var)

        # propagation
        x_const = self.propagate(scaled_hyperedge_index, size=(num_nodes, num_edges), x=x_var, coef=scaled_coef)
        # sum_rhs = torch.sum(rhs).view(1, 1) / num_edges
        sum_rhs = torch.sum(rhs).view(1, 1)
        sum_rhs = self.rhs_emb(sum_rhs)
        # aggr_const = x_const.sum(axis=0)
        aggr_const = self.lin_c(x_const.mean(axis=0))
        rhs_ = sum_rhs - aggr_const
        # rhs_ = rhs_ * inv_s_v.view(-1, 1)
        rhs_ = rhs_ * self.rhs_emb(inv_s_v.view(-1, 1))
        # rhs_ = self.lin_c(rhs_)   

        x_var = rhs_ + x_var
        return x_var, x_const, emb_rhs

    """
    # @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        hyperedge_index: Tensor,
        *,
        coef: Tensor,
        rhs: Tensor,
        x_var: Optional[Tensor] = None,
        hyperedge_weight: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        # x = self.lin_x(x)
        num_nodes = int(hyperedge_index[0].max()) + 1
        # num_nodes = x.size(0)
        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = hyperedge_index.new_ones(num_edges)  # It can be used to assign the obj function higher importance.
        hyperedge_index, coef = torch_sparse.coalesce(hyperedge_index, coef, num_nodes, num_edges, op="mean")

        # Column(constraints) scaling
        s_c = scatter(coef.abs(), hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        inv_s_c = 1.0 / s_c
        inv_s_c[inv_s_c == float("inf")] = 0
        # print(f"max inv_s_c: {inv_s_c.max()}")

        # # Column(constraints) normalization
        # deg_c = scatter(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        # inv_deg_c = 1.0 / deg_c
        # inv_deg_c[inv_deg_c == float("inf")] = 0
        # # inv_s_c *= inv_deg_c

        # Scaled coefficient and RHS
        # diag_idx = torch.tensor([range(len(inv_s_c)), range(len(inv_s_c))], device=inv_s_c.device, dtype=torch.long)
        # diag_idx, inv_s_c = torch_sparse.coalesce(diag_idx, inv_s_c, num_edges, num_edges, op="mean")
        # scaled_hyperedge_index, scaled_coef = torch_sparse.spspmm(hyperedge_index, coef, diag_idx, inv_s_c, num_nodes, num_edges, num_edges)
        # torch.sparse_coo_tensor(scaled_hyperedge_index, scaled_coef).to_dense()
        # rhs = inv_s_c.view(-1, 1) * rhs.view(-1, 1)
        emb_rhs = self.rhs_emb(rhs)

        # Row(variables) scaling
        s_v = scatter(coef.abs(), hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        inv_s_v = 1.0 / s_v
        inv_s_v[inv_s_v == float("inf")] = 0
        # print(f"max inv_s_v: {inv_s_v.max()}")

        if x_var is None:
            x_var = self.propagate(hyperedge_index.flip([0]), size=(num_edges, num_nodes), x=emb_rhs, coef=coef, scaling=inv_s_v)

        # propagation
        x_const = self.propagate(hyperedge_index, size=(num_nodes, num_edges), x=x_var, coef=coef, scaling=inv_s_c)
        if self.org_version:
            x = x_var.clone().detach()
            x_const = emb_rhs - x_const
        if self.is_lin_c:
            x_const = self.lin_c(x_const)
            # x_const = self.layer_norm(x_const)
            # x_const = self.activation(x_const)
        x_var = self.propagate(hyperedge_index.flip([0]), size=(num_edges, num_nodes), x=x_const, coef=coef, scaling=inv_s_v)

        # if self.weight_l:
        #     x_var = self.lin_l(x_var)
        # if self.weight_r:
        #     x = self.lin_r(x)
        if self.org_version:
            x_var = x_var + x
        return x_var, x_const, emb_rhs
    """

    def message(self, x_j: Tensor, coef: Optional[Tensor] = None, scaling_i: Optional[Tensor] = None) -> Tensor:
        out = x_j.view(-1, self.out_channels)

        if coef is not None:
            out = coef.view(-1, 1) * out

        if scaling_i is not None:
            out = scaling_i.view(-1, 1) * out

        return out


# for debugging code
# batch = batch.to(args.device)
# hyperedge_index = batch.edge_index
# coef = batch.coef
# rhs = batch.rhs
# x_var = torch.eye(4, device=hyperedge_index.device) * (torch.tensor(range(4), device=hyperedge_index.device).reshape(-1, 1) + 1)
# x = torch.empty(hyperedge_index[0].max() + 1, args.hidden_size, requires_grad=False, device=args.device)
# x_var = torch.nn.init.xavier_uniform_(x)
# rhs_emb = Sequential(
#              PeriodicEmbeddings(args.n_rhs_features, args.emb_size, lite=False),
#              torch.nn.Flatten(),
#              MLP(d_in=args.emb_size * args.n_rhs_features, **mlp_config),
#          )
# ------------------------------------------
# hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]], dtype=torch.long, device="cuda:0")
# coef = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32, device="cuda:0")
# coef = torch.tensor([134, 2, 2, 23, 5, 6], dtype=torch.float32, device="cuda:0")
# rhs = torch.tensor([[10], [100]], dtype=torch.float32, device="cuda:0")
# x_var = torch.tensor([[1, 33, 22, 11],[23, 1, 44, 51], [13, 22, 3, 0], [68, 9, 0, 2]], dtype=torch.float32, device="cuda:0")
# num_edges = 2
# num_nodes = 4

# conv = MILPConv(rhs_emb, args.hidden_size, args.hidden_size, weight_l=False, weight_r=False).to(args.device)
# x_var, x_const, _ = conv(hyperedge_index, coef=coef, rhs=rhs)


class GraphDataset_HG(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, args, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.args = args
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        # filepath = sample_files[0]
        GraphFilepath, solFilePath = filepath
        with open(GraphFilepath, "rb") as f:
            graphData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        varNames = solData["var_names"]

        sols = solData["sols"][: self.args.n_sols]  # [0:300]
        objs = solData["objs"][: self.args.n_sols]  # [0:300]

        # sols = np.round(sols, 0)
        return graphData, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        graphData, sols, objs, varNames = self.process_sample(self.sample_files[index])
        A, B, v_map, target_vars, sense, coef_obj = graphData
        # if not self.args.include_obj:
        #     A = coo_array(A.todense()[1:])
        #     B = coo_array(B.todense()[1:])

        edge_indices = torch.tensor(np.array([A.col, A.row]), dtype=torch.long)
        coef = torch.tensor(A.data, dtype=torch.float32)
        # rhs = torch.tensor(B, dtype=torch.int32)
        # rhs = torch.tensor(B.todense(), dtype=torch.int32)
        graph = HypergraphNodeData(edge_indices, coef)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.rhs = torch.tensor(np.concatenate([sense, B], axis=1), dtype=torch.float32)
        graph.coef_obj = torch.tensor(coef_obj, dtype=torch.float32).reshape(-1)
        graph.num_nodes = len(v_map)
        graph.solutions = torch.tensor(sols, dtype=torch.float32).reshape(-1)
        graph.objVals = torch.tensor(objs, dtype=torch.float32)
        # graph.solutions = torch.FloatTensor(sols).reshape(-1)
        # graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.varNames = varNames

        varname_dict = {name: i for i, name in enumerate(varNames)}
        varname_map = torch.tensor([varname_dict[v] for v in v_map], dtype=torch.long)
        target_vars = torch.tensor(target_vars, dtype=torch.long)
        graph.varInds = [[varname_map], [target_vars]]
        return graph


# varNames = ['d', 'e', 'f', 'c', 'b', 'a']
# graph_varNames = sorted(varNames)
# v_map = {name: i for i, name in enumerate(graph_varNames)}


class HypergraphNodeData(torch_geometric.data.Data):
    def __init__(self, edge_indices, coef):
        super().__init__()
        self.edge_index = edge_indices
        self.coef = coef

    def __inc__(self, key, value, store, *args, **kwargs):
        """ """
        if key == "edge_index":
            return torch.tensor([[self.edge_index[0].max() + 1], [self.edge_index[1].max() + 1]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        # CONSTRAINT EMBEDDING
        self.cons_embedding = Sequential(
            LayerNorm(cons_nfeats),
            Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = Sequential(
            LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = Sequential(
            LayerNorm(var_nfeats),
            Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = Sequential(
            Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, 1, bias=False),
        )

        output_module = []
        output_module.append(Linear(emb_size, emb_size))
        output_module.append(torch.nn.ReLU())
        output_module.append(Linear(emb_size, 1))

        if ds2type[args.task_name] == "B":
            output_module.append(torch.nn.Sigmoid())
        self.output_module = Sequential(*output_module)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        constraint_features = self.conv_v_to_c2(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v2(constraint_features, edge_indices, edge_features, variable_features)

        # output = self.output_module(variable_features).squeeze(-1)
        # A final MLP on the variable features
        output = self.output_module(variable_features)

        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__(aggr="add")
        emb_size = 64

        self.feature_module_left = Sequential(Linear(emb_size, emb_size))
        self.feature_module_edge = Sequential(Linear(1, emb_size, bias=False))
        self.feature_module_right = Sequential(Linear(emb_size, emb_size, bias=False))
        self.feature_module_final = Sequential(
            LayerNorm(emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
        )

        self.post_conv_module = Sequential(LayerNorm(emb_size))

        # output_layers
        self.output_module = Sequential(
            Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        # b = torch.cat([self.post_conv_module(output), right_features], dim=-1)
        # a = self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i) + self.feature_module_edge(edge_features) + self.feature_module_right(node_features_j)
        )

        return output


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, args, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.args = args
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        # filepath = train_files[0]
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            # bgData = pickle.load(f)
            bgData = CPU_Unpickler(f).load()

        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData["var_names"]

        sols = solData["sols"][: self.args.n_sols]  # [0:300]
        objs = solData["objs"][: self.args.n_sols]  # [0:300]

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars = BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        constraint_features[torch.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map, dtype=torch.long)
        if isinstance(b_vars, torch.Tensor):
            b_vars = b_vars.to(dtype=torch.long)
        else:
            b_vars = torch.tensor(b_vars, dtype=torch.long)

        graph.varInds = [[varname_map], [b_vars]]

        return graph


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy_position(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 18

        # CONSTRAINT EMBEDDING
        self.cons_embedding = Sequential(
            LayerNorm(cons_nfeats),
            Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = Sequential(
            LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = Sequential(
            LayerNorm(var_nfeats),
            Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        # self.output_module = Sequential(
        #     Linear(emb_size, emb_size),
        #     torch.nn.ReLU(),
        #     Linear(emb_size, 1, bias=False),
        # )
        output_module = []
        output_module.append(Linear(emb_size, emb_size))
        output_module.append(torch.nn.ReLU())
        output_module.append(Linear(emb_size, 1))

        if ds2type[args.task_name] == "B":
            output_module.append(torch.nn.Sigmoid())
        self.output_module = Sequential(*output_module)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        constraint_features = self.conv_v_to_c2(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v2(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        # output = self.output_module(variable_features).squeeze(-1)
        output = self.output_module(variable_features)

        return output


class GraphDataset_position(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, args, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.args = args
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            # bgData = pickle.load(f)
            bgData = CPU_Unpickler(f).load()
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData["var_names"]

        sols = solData["sols"][: self.args.n_sols]  # [0:300]
        objs = solData["objs"][: self.args.n_sols]  # [0:300]

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars = BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        # lens = variable_features.shape[0]
        # feature_widh = 12  # max length 4095
        # position = torch.arange(0, lens, 1)

        # DEVICE = variable_features.device
        # position_feature = torch.zeros(lens, feature_widh).to(DEVICE)
        # for i in range(len(position_feature)):
        #     binary = str(bin(position[i]).replace("0b", ""))

        #     for j in range(len(binary)):
        #         position_feature[i][j] = int(binary[-(j + 1)])

        # v = torch.concat([variable_features, position_feature], dim=1)

        # variable_features = v
        variable_features = postion_get(variable_features)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map)
        if torch.is_tensor(b_vars):
            b_vars = b_vars.type(torch.long)
        else:
            b_vars = torch.tensor(b_vars, dtype=torch.long)

        graph.varInds = [[varname_map], [b_vars]]

        return graph


def postion_get(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 12  # max length 4095
    position = torch.arange(0, lens, 1)

    DEVICE = variable_features.device
    position_feature = torch.zeros(lens, feature_widh).to(DEVICE)
    for i in range(len(position_feature)):
        binary = str(bin(position[i]).replace("0b", ""))

        for j in range(len(binary)):
            position_feature[i][j] = int(binary[-(j + 1)])

    v = torch.concat([variable_features, position_feature], dim=1)

    return v


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
            return super().find_class(module, name)
