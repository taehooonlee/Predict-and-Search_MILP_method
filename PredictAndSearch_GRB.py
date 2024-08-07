import argparse
import os
import random

import gurobipy
import numpy as np
import torch
from gurobipy import GRB

from helper import get_a_new2

# 4 public datasets, IS, WA, CA, IP
dev_num = "7"
TaskName = "CJ"
# test_set = "train"
test_set = "test"
DEVICE = torch.device(f"cuda:{dev_num}" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def test_hyperparam(task):
    """
    set the hyperparams
    k_0, k_1, delta
    """
    if task == "IP":
        return 400, 5, 1
    elif task == "IS":
        return 300, 300, 15
    elif task == "CJ":
        return 10, 60, 3
    elif task == "WA":
        return 0, 600, 5
    elif task == "CA":
        return 400, 0, 10


k_0, k_1, delta = test_hyperparam(TaskName)

# set log folder
solver = "GRB"
test_task = f"{solver}/{k_0}_{k_1}_{delta}/{test_set}"
base_dir = "/mnt/disk1/thlee/MILP/pas"
log_dir = os.path.join(base_dir, "test_logs", TaskName, test_task)
model_dir = os.path.join(base_dir, f"pretrain/{TaskName}_train/{TaskName}_trainmodel_best.pth")
ins_dir = os.path.join(base_dir, f"instance/{TaskName}/{test_set}")
print(f"\nlog dir: {log_dir}\n")

if not os.path.isdir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# load pretrained model
if TaskName == "IP":
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy
    from GCN import postion_get
else:
    from GCN import GNNPolicy

policy = GNNPolicy().to(DEVICE)
state = torch.load(model_dir, map_location=torch.device(DEVICE))
policy.load_state_dict(state)

ins_names = sorted(os.listdir(ins_dir))
for ins in ins_names:
    ins_name_to_read = os.path.join(ins_dir, ins)

    # get bipartite graph as input
    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
    constraint_features = c_nodes.cpu()
    constraint_features[np.isnan(constraint_features)] = 1  # remove nan value
    variable_features = v_nodes
    if TaskName == "IP":
        variable_features = postion_get(variable_features)
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features = torch.ones(edge_features.shape)

    # prediction
    BD = (
        policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        )
        .sigmoid()
        .cpu()
        .squeeze()
    )

    # align the variable name betweend the output and the solver
    all_varname = []
    for name in v_map:
        all_varname.append(name)
    binary_name = [all_varname[i] for i in b_vars]
    scores = []  # get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type = "C"
        if all_varname[i] in binary_name:
            type = "BINARY"
        scores.append([i, all_varname[i], BD[i].item(), -1, type])

    scores.sort(key=lambda x: x[2], reverse=True)

    scores = [x for x in scores if x[4] == "BINARY"]  # get binary

    fixer = 0
    # fixing variable picked by confidence scores
    count1 = 0
    for i in range(len(scores)):
        if count1 < k_1:
            scores[i][3] = 1
            count1 += 1
            fixer += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1

    print(f"instance: {ins}, " f"fix {k_0} 0s and " f"fix {k_1} 1s, delta {delta}. ")

    # read instance
    gurobipy.setParam("LogToConsole", 1)  # hideout
    m = gurobipy.read(ins_name_to_read)
    m.Params.TimeLimit = 600
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = f"{log_dir}/{ins.split('.')[0]}.log"

    # trust region method implemented by adding constraints
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        # tmp_var = m1.addVar(f'alp_{tar_var}', 'C')
        tmp_var = m.addVar(name=f"alp_{tar_var}", vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f"alpha_up_{i}")
        m.addConstr(tmp_var >= x_star - tar_var, name=f"alpha_dowm_{i}")
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    m.optimize()
