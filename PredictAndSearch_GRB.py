import gc
import math
import multiprocessing as mp
import os
import random
import time
from multiprocessing import Process, Queue
from os.path import join as pjoin

import gurobipy
import numpy as np
import torch
from gurobipy import GRB
from numba import cuda

from helper import create_parser, ds2type, get_a_new2, get_HG_from_GRB

# 4 public datasets, IS, WA, CA, IP


def test_hyperparam(task):
    """
    set the hyperparams
    k_0, k_1, delta
    """
    if task == "IP":
        # return 400, 5, 1
        return 100, 5, 15
    elif task == "WA":
        # return 0, 600, 5
        # return 0, 600, 20
        # return 0, 600, 40
        # return 0, 600, 60
        return 0, 600, 100
    elif task == "IS":
        return 300, 300, 15
    elif task == "CJ":
        return 10, 60, 3
    elif task == "CA":
        return 400, 0, 10


def prediction(args, ins_name_to_read, model_path):
    # load pretrained model
    if args.graph_type == "BG":
        if args.task_name == "IP":
            # Add position embedding for IP model, due to the strong symmetry
            from GCN import GNNPolicy_position as GNNPolicy
            from GCN import postion_get
        else:
            from GCN import GNNPolicy
    elif args.graph_type == "HG":
        from GCN import GNNPolicy_MILP as GNNPolicy

        # A, B, v_map, target_vars = get_HG_from_GRB(args, ins_name_to_read)
        # args.num_vars = len(v_map)

    print(f"Device: {args.device}")
    torch.cuda.set_device(args.device)
    cuda.select_device(args.device.index)  # choosing GPU
    state = torch.load(model_path, map_location=torch.device(args.device))
    model = GNNPolicy(args).to(args.device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        if args.graph_type == "BG":
            A, v_map, var_nodes, const_nodes, target_vars = get_a_new2(ins_name_to_read)
            const_nodes[torch.isnan(const_nodes)] = 1  # remove nan value
            var_nodes = postion_get(var_nodes) if args.task_name == "IP" else var_nodes
            const_nodes = const_nodes.to(args.device)
            var_nodes = var_nodes.to(args.device)
            edge_indices = A._indices().to(args.device)
            edge_features = A._values().unsqueeze(1)
            edge_features = torch.ones(edge_features.shape).to(args.device)

            output = model(const_nodes, edge_indices, edge_features, var_nodes)  # prediction
            del const_nodes, var_nodes, edge_indices, edge_features, A

        elif args.graph_type == "HG":
            A, B, v_map, target_vars = get_HG_from_GRB(args, ins_name_to_read)
            edge_indices = torch.tensor(np.array([A.col, A.row]), dtype=torch.long).to(args.device)
            coef = torch.tensor(A.data, dtype=torch.float32).to(args.device)
            rhs = torch.tensor(B.todense(), dtype=torch.int32).to(args.device)

            output = model(edge_indices, coef, rhs)  # prediction
            del edge_indices, coef, rhs, A

        if ds2type[args.task_name] == "B":
            output = output.sigmoid()
    if args.evi_loss and ds2type[args.task_name] == "I":
        mu = output[0].cpu().squeeze().numpy().copy()
        v = output[1].cpu().squeeze().numpy().copy()
        alpha = output[2].cpu().squeeze().numpy().copy()
        beta = output[3].cpu().squeeze().numpy().copy()
        output_ = (mu, v, alpha, beta)
    else:
        output_ = output.cpu().squeeze().numpy().copy()
    # target_vars_ = target_vars.cpu().numpy().copy()

    del output, state, model
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            del obj  # Delete the GPU tensor
    gc.collect()
    torch.cuda.empty_cache()
    cuda.select_device(args.device.index)  # choosing GPU
    device = cuda.get_current_device()
    print(f"Reset device: {device}")
    cuda.close()

    return output_, v_map, target_vars


def get_target_vars_score(args, pred, v_map, target_vars):
    # align the variable name between the output and the solver
    if args.evi_loss and ds2type[args.task_name] == "I":
        mu, v, alpha, beta = pred
    all_varname = list(v_map)
    target_var_name = [all_varname[i] for i in target_vars]
    scores = []  # get a list of (index, VariableName, Prob, is_fixed, is_target, Pred)
    for i, var_name in enumerate(v_map):
        is_target = True if var_name in target_var_name else False
        if args.evi_loss and ds2type[args.task_name] == "I":
            std = math.sqrt(beta[i] / (v[i] * (alpha[i] - 1)))
            scores.append([i, var_name, std, -1, is_target, mu[i]])
        else:
            scores.append([i, var_name, pred[i].item(), -1, is_target])
    scores.sort(key=lambda x: x[2], reverse=True)
    scores = [x for x in scores if x[4]]  # get target variables
    return scores


def fix_vars_by_score(args, ins, scores, k_0, k_1):
    # fixing variable picked by confidence scores
    count0, count1 = 0, 0
    for sc in scores:
        if count1 == k_1:
            break
        sc[3] = 1
        count1 += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    for sc in scores:
        if count0 == k_0:
            break
        sc[3] = 0
        count0 += 1
    print(f"instance: {ins}, fix {k_0} 0s, fix {k_1} 1s, and total {count0+count1} fixed. ")
    return scores


def read_instance(ins_name_to_read, log_path):
    # read instance
    gurobipy.setParam("LogToConsole", 0)  # hideout
    m = gurobipy.read(ins_name_to_read)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = log_path
    return m


def optimize_w_delta(args, m, scores, delta):
    # trust region method implemented by adding constraints
    instance_variables = m.getVars()
    instance_variables.sort(key=lambda v: v.VarName)
    variabels_map = {v.VarName: v for v in instance_variables}

    all_tmp = 0
    for i, sc in enumerate(scores):
        x_pred = sc[3]  # 1,0,-1, decide whether need to fix
        if x_pred == -1:
            continue
        tar_var = variabels_map[sc[1]]  # target variable <-- variable map
        tmp_var = m.addVar(name=f"alp_{tar_var}", vtype=GRB.CONTINUOUS)
        all_tmp += tmp_var
        m.addConstr(tmp_var >= tar_var - x_pred, name=f"alpha_up_{i}")
        m.addConstr(tmp_var >= x_pred - tar_var, name=f"alpha_dowm_{i}")
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    m.optimize()


def pred_and_search(args, q, ins_dir, log_dir, model_path):
    # ins = ins_names[1]
    k_0, k_1, delta = test_hyperparam(args.task_name)
    while True:
        ins = q.get()
        if ins is None:
            break

        log_path = f"{log_dir}/{ins.split('.')[0]}.log"
        if os.path.exists(log_path):
            continue

        ins_name_to_read = pjoin(ins_dir, ins)
        pred, v_map, target_vars = prediction(args, ins_name_to_read, model_path)
        scores = get_target_vars_score(args, pred, v_map, target_vars)
        scores = fix_vars_by_score(args, ins, scores, k_0, k_1)
        m = read_instance(ins_name_to_read, log_path)
        optimize_w_delta(args, m, scores, delta)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    mp.set_start_method("spawn")
    target_var = {"IP": "B", "WA": "B", "CJ": "I"}

    # create parser
    args = create_parser()

    # args.exp_id = "240831-210103"
    # args.task_name = "CJ"
    # args.graph_type = "HG"
    # args.init_x = "uniform"
    # args.evi_loss = True
    # args.n_conv = 2
    # args.device = 3

    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.var_type = target_var[args.task_name]
    k_0, k_1, delta = test_hyperparam(args.task_name)

    # set folder
    solver = "GRB"
    test_task = f"{solver}/{k_0}_{k_1}_{delta}"
    ins_dir = pjoin(args.dir_base, "instance", args.task_name, "test")
    log_dir = pjoin(args.dir_base, "test_logs", args.task_name, args.exp_id, test_task)
    model_path = pjoin(args.dir_base, "pretrain", f"{args.task_name}_train_{args.graph_type}", args.exp_id, "model_best.pth")
    print(f"\nlog dir: {log_dir}\n")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # predict and search using multiple workers
    ps = []
    q = Queue()
    ins_names = sorted(os.listdir(ins_dir))
    ins_names = [file for file in ins_names if not file.endswith(".json")]
    for ins in ins_names:
        q.put(ins)  # add ins
    for _ in range(args.n_workers):
        q.put(None)  # add stop signal
    for _ in range(args.n_workers):
        p = Process(target=pred_and_search, args=(args, q, ins_dir, log_dir, model_path))  # run PAS
        p.start()
        ps.append(p)
        time.sleep(3)
    for p in ps:
        p.join()
    print("done")
    print("done")
