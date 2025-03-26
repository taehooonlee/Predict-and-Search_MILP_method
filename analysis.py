import shutil

args.task_name = "WA"
args.graph_type = "BG"

ds = "WA/train"
INS_DIR_ORG = os.path.join(args.dir_base, f"instance/WA/original/train")
INS_DIR_NEW = os.path.join(args.dir_base, f"instance/{ds}")

dir_graph_org = pjoin(args.dir_base, f"dataset/{args.task_name}/original/train_backup/{args.graph_type}")
dir_sol_org = pjoin(args.dir_base, f"dataset/{args.task_name}/original/train_backup/solution")
dir_log_org = pjoin(args.dir_base, f"dataset/{args.task_name}/original/train_backup/logs")

dir_graph_new = pjoin(args.dir_base, f"dataset/{args.task_name}/train/{args.graph_type}")
dir_sol_new = pjoin(args.dir_base, f"dataset/{args.task_name}/train/solution")
dir_log_new = pjoin(args.dir_base, f"dataset/{args.task_name}/train/logs")

sample_names = os.listdir(dir_sol_org)
len(sample_names)
sample_files = [
    (name, pjoin(dir_sol_org, name), pjoin(dir_graph_org, name).replace("sol", "bg"), pjoin(dir_log_org, name).replace("sol", "log"))
    for name in sample_names
]


random.shuffle(sample_files)
train_names = sample_names[:53]
len(train_names)
test_names = sample_names[2000:2400]
len(test_names)


for name, sol_, bg_, log_ in sample_files:
    if name in train_names:
        shutil.move(sol_, dir_sol_new)
        shutil.move(bg_, dir_graph_new)
        shutil.move(log_, dir_log_new)
        shutil.copy(pjoin(INS_DIR_ORG, name[:-4]), INS_DIR_NEW)
    # elif name in test_names:
    #     shutil.move(sol_, dir_sol_new.replace("/train/", "/test/"))
    #     shutil.move(bg_, dir_graph_new.replace("/train/", "/test/"))
    #     shutil.move(log_, dir_log_new.replace("/train/", "/test/"))
    #     shutil.copy(pjoin(INS_DIR_ORG, name[:-4]), INS_DIR_NEW.replace("/train", "/test"))

import os

sol_names = os.listdir("/mnt/disk1/thlee/MILP/pas/dataset/WA/train/solution")
log_names = os.listdir("/mnt/disk1/thlee/MILP/pas/dataset/WA/train/logs")
log_dir = "/mnt/disk1/thlee/MILP/pas/dataset/WA/train/logs"

sol_names = [n.split(".")[0] for n in sol_names]
log_names = [n.split(".")[0] for n in log_names]


for log in log_names:
    if log not in sol_names:
        print(log)
        os.remove(os.path.join(log_dir, f"{log}.mps.gz.log"))


# ------------------------------------------------------------------------------------------------------------------------------
# count trainable parameter and FLOPs
# ------------------------------------------------------------------------------------------------------------------------------

import random
from os.path import join as pjoin

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

import helper

args = helper.create_parser()
args.device = "4"
args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

# ------- BG -------
args.graph_type = "BG"
args.activation = "relu"

# ------- HG -------
args.graph_type = "HG"
args.weight_l = False
args.weight_r = False
args.num_vars = 1083
args.n_conv = 4
args.task_name = "CJ"

if args.graph_type == "BG":
    if args.task_name == "IP":
        # Add position embedding for IP model, due to the strong symmetry
        from GCN import GNNPolicy_position as GNNPolicy
        from GCN import GraphDataset_position as GraphDataset
    else:
        from GCN import GNNPolicy, GraphDataset
elif args.graph_type == "HG":
    from GCN import GNNPolicy_MILP as GNNPolicy
    from GCN import GraphDataset_HG as GraphDataset

PredictModel = GNNPolicy(args).to(args.device)

# Get the number of trainable parameters
num_trainable_params = sum(p.numel() for p in PredictModel.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")


# dir_graph = pjoin(args.dir_base, f"dataset/{args.task_name}/train/{args.graph_type}")
# dir_sol = pjoin(args.dir_base, f"dataset/{args.task_name}/train/solution")
# sample_names = os.listdir(dir_sol)
# sample_files = [(pjoin(dir_graph, name).replace("sol", args.graph_type.lower()), pjoin(dir_sol, name)) for name in sample_names]
# random.shuffle(sample_files)
# train_files = sample_files[: int(0.80 * len(sample_files))]
# valid_files = sample_files[int(0.80 * len(sample_files)) :]
# train_data = GraphDataset(train_files)
# batch = train_data[0].to(args.device)

# input_tensor = (batch.constraint_features.detach(), batch.edge_index.detach(), batch.edge_attr.detach(), batch.variable_features.detach())

# input_tensor = (batch.edge_index.detach(), batch.coef.detach(), batch.rhs.detach())

# flops = FlopCountAnalysis(PredictModel.eval(), input_tensor)
# print(flop_count_table(flops))
# flops.total()

# flops.total()


import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("mip1")

# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")
a = m.addVar(vtype=GRB.BINARY, name="a")

# Set objective
m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
# m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
constr = gp.LinExpr()
constr += x
constr += 2 * y
constr += 3 * z
constr
m.addConstr(constr <= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")
m.update()
m.optimize()

# m
# mvars = m.getVars()
# mvars.sort(key=lambda v: v.VarName)
# A = m.getA()
# A.todense()

m = gp.Model("mip2")

# Create variables
z = m.addVar(vtype=GRB.BINARY, name="z")
y = m.addVar(vtype=GRB.BINARY, name="y")
x = m.addVar(vtype=GRB.BINARY, name="x")

# Set objective
m.setObjective(-x - y - 2 * z, GRB.MINIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
# m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
constr = gp.LinExpr()
constr += x
constr += 2 * y
constr += 3 * z
constr
m.addConstr(constr <= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(-x - y <= -1, "c1")
m.update()
m.optimize()
m.optimize()


import hypernetx as hnx

# Define a hypergraph with hyperedges
edges = {"e1": [1, 2, 3], "e2": [3, 4], "e3": [4, 5, 6], "e4": [5, 7]}

# Create a Hypergraph object
H = hnx.Hypergraph(edges)

# Display the hypergraph structure
print(H.incidence_dict)
adj = H.adjacency_matrix().todense()


import networkx as nx

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3)])

# Compute the shortest paths between all pairs of nodes
all_paths = dict(nx.all_pairs_shortest_path(G))

# Print shortest paths for each source node
for source, paths in all_paths.items():
    print(f"Shortest paths from node {source}:")
    for target, path in paths.items():
        print(f"  to {target}: {path}")


# Define the hypergraph edges
edges = {"e1": [1, 2, 3], "e2": [3, 4], "e3": [5, 6], "e4": [5, 7]}

import numpy as np


def len_shortest_paths(edges):
    # Create the hypergraph using HyperNetX
    H = hnx.Hypergraph(edges)

    # Initialize an empty NetworkX graph
    G = nx.Graph()

    # For each hyperedge in the hypergraph, connect all pairs of nodes within that hyperedge
    for edge, nodes in H.incidence_dict.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
    print("graph has been built")
    # Print the adjacency list of the resulting graph
    # for node, neighbors in G.adjacency():
    #     print(f"Node {node} is connected to {list(neighbors)}")

    # Compute the shortest paths between all pairs of nodes
    all_paths = dict(nx.all_pairs_shortest_path(G))
    print("Computed the shortest paths")
    # Print shortest paths for each source node
    len_paths = []
    for source, paths in all_paths.items():
        # print(f"Shortest paths from node {source}:")
        for target, path in paths.items():
            # print(f"  to {target}: {path}")
            len_path = len(path) - 1
            if len_path >= 1:
                len_paths.append(len_path)
    print("returns the list of len_paths")
    return len_paths


len_paths = []
for batch in tqdm(train_loader):
    # break
    edge_index = np.array(batch.edge_index.T)
    edges = {}
    for n_idx, e_idx in edge_index:
        e_name = f"e{e_idx}"
        if edges.get(e_name) is None:
            edges[e_name] = []
        edges[e_name].append(n_idx)
    print("edges have been built")
    len_paths += len_shortest_paths(edges)

print(f"max: {np.max(len_paths)}")
print(f"min: {np.min(len_paths)}")
print(f"mean: {np.mean(len_paths)}")
print(f"std: {np.std(len_paths)}")


prev_coef_A = None
for i, filename in enumerate(filenames):
    filepath = os.path.join(ins_dir, filename)
    m = gp.read(filepath)
    coef_A = m.getA().todense()
    print(f"iter: {i}")

    coef_A = np.array(coef_A == 0)

    # if (prev_coef_A is not None) and (((prev_coef_A == 0) != (coef_A == 0)).sum() != 0):
    if (prev_coef_A is not None) and ((prev_coef_A != coef_A).sum() != 0):
        print(i)
        print(filepath)
        break
    prev_coef_A = coef_A.copy()

np.all(prev_coef_A == coef_A)


coef_A = m.getA().todense()
((coef_A != 0).sum(axis=1) / coef_A.shape[1]).mean() * 100
(coef_A != 0).sum() / (coef_A.shape[0] * coef_A.shape[1]) * 100

(coef_A != 0).sum(axis=1).max()

constraints = m.getConstrs()

# Retrieve and print the names of all constraints
for constr in constraints:
    print(constr.ConstrName)
