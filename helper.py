import argparse
import os
import pathlib
import pickle
import random
import sys
import time

import gurobipy as gp
import numpy as np
import pyscipopt as scp
import torch
import torch.nn as nn
from gurobipy import GRB
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def position_get_ordered_flt(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 20  # max length 4095
    sorter = variable_features[:, 1]
    position = torch.argsort(sorter)
    position = position / float(lens)

    position_feature = torch.zeros(lens, feature_widh)

    for row in range(position.shape[0]):
        flt_indx = position[row]
        divider = 1.0
        for ks in range(feature_widh):
            if divider <= flt_indx:
                position_feature[row][ks] = 1
                flt_indx -= divider
            divider /= 2.0
        # print(row,position[row],position_feature[row])
    position_feature = position_feature.to(device)
    variable_features = variable_features.to(device)
    v = torch.concat([variable_features, position_feature], dim=1)
    return v


def get_BG_from_scip(ins_name):
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == "BINARY":
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    # for i in range(len(cons_map)):
    #   tmp=0
    #   coeff=m.getValsLinear(cons_map[i][0])
    #    for k in coeff:
    #        tmp+=coeff[k]
    #    cons_map[i].append(tmp)
    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]
    # print(cons)
    # quit()
    A = []
    for i in range(ncons):
        A.append([])
        for j in range(nvars + 2):
            A[i].append(0)
    A.append(obj_cons)
    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        A[cind][-2] = rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            A[cind][v_indx] = 1
            A[cind][-1] += 1
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            if v_indx == 1066:
                print(coeff[k], lcons)
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = np.array(A, dtype=np.float32)

    A = A[:, :-2]
    A = torch.as_tensor(A).to(device).to_sparse()
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    v_nodes = position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_BG_from_GRB(ins_name):
    # vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]

    m = gp.read(ins_name)
    ori_start = 6
    emb_num = 15

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.VarName)

    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.VarName] = indx

    nvars = len(mvars)

    v_nodes = []
    b_vars = []
    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e20
        # tp=[0,0,0,0,0]
        if mvars[i].VType == "B":
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    obj_node = [0, 0, 0, 0]

    nobjs = obj.size()
    for i in range(nobjs):
        vnm = obj.getVar(i).VarName
        v = obj.getCoeff(i)
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        v_nodes[v_indx][0] = v
        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]

    cons = m.getConstrs()
    ncons = len(cons)
    lcons = ncons
    c_nodes = []

    A = []
    for i in range(ncons):
        A.append([])
        for j in range(nvars + 2):
            A[i].append(0)
    A.append(obj_cons)
    for i in range(ncons):
        tmp_v = []
        tmp_c = []

        sense = cons[i].Sense
        rhs = cons[i].RHS
        nzs = 0

        if sense == "<":
            sense = 0
        elif sense == ">":
            sense = 1
        elif sense == "=":
            sense = 2

        tmp_c = [0, 0, rhs, sense]
        summation = 0
        tmp_v = [0, 0, 0, 0, 0]
        for v in mvars:
            v_indx = v_map[v.VarName]
            ce = m.getCoeff(cons[i], v)

            if ce != 0:
                nzs += 1
                summation += ce
                A[i][v_indx] = 1
                A[i][-1] += 1

        if nzs == 0:
            continue
        tmp_c[0] = summation / nzs
        tmp_c[1] = nzs
        c_nodes.append(tmp_c)
        for v in mvars:
            v_indx = v_map[v.VarName]
            ce = m.getCoeff(cons[i], v)

            if ce != 0:
                v_nodes[v_indx][2] += 1
                v_nodes[v_indx][1] += ce / lcons
                v_nodes[v_indx][3] = max(v_nodes[v_indx][3], ce)
                v_nodes[v_indx][4] = min(v_nodes[v_indx][4], ce)

    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = np.array(A, dtype=np.float32)

    A = A[:, :-2]
    A = torch.as_tensor(A).to(device).to_sparse()
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    v_nodes = position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_a_new2(ins_name):
    # if ins_name.split("/")[-3] == "CJ":
    #     d_num = int(ins_name.split("/")[-1].split(".")[0][-2:]) % 4
    # else:
    #     d_num = int(ins_name.split("_")[-1].split(".")[0]) % 4
    # device = torch.device(f"cuda:{d_num}" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    epsilon = 1e-5

    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)
    print(f"insname: {ins_name}")

    ncons = m.getNConss()
    nvars = m.getNVars()
    print(f"nvars: {nvars}")

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == "BINARY":
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            # values_spr.append(v)
            values_spr.append(1)
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    c_nodes.append(obj_node)
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        # A[cind][-2]=rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            # A[cind][v_indx]=1
            # A[cind][-1]+=1
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, epsilon, 1)
    # v_nodes=position_get_ordered(v_nodes)
    # v_nodes=position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, epsilon, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_HG_from_GRB(args, ins_name):
    m = gp.read(ins_name)

    # mapping each variable's name to index
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.VarName)
    v_map = {v.VarName: idx for idx, v in enumerate(mvars)}

    # variables that have the specified type
    target_vars = [idx for idx, v in enumerate(mvars) if v.VType == args.var_type]

    # making all constraints to have the inequity(<=)
    A = m.getA()
    B = np.array(m.getAttr("RHS", m.getConstrs()))
    sense = np.array(m.getAttr("Sense", m.getConstrs()))
    index_ge = np.where(sense == ">")[0]
    A[index_ge] *= -1
    B[index_ge] *= -1

    # appending the objective function to the constraints matrix
    coef_obj = m.getAttr("Obj", mvars)
    model_sense = m.getAttr(GRB.Attr.ModelSense)
    if model_sense == GRB.MAXIMIZE:
        coef_obj = np.array(coef_obj) * -1
    A = vstack([coef_obj, A])
    B = vstack([np.zeros([1, 1]), B.reshape([-1, 1])])
    # C = None

    # scalling all constraints by the maximum absolute coefficient in each constraint
    # if args.scaling:
    #     AB = hstack([A, B])
    #     max_values = np.absolute(AB).max(axis=1).todense()
    #     inv_max_values = 1.0 / (max_values + 1e-5)
    #     A = A.multiply(inv_max_values)
    # B = B.multiply(inv_max_values)
    # C = 1.0 / (A.sum(axis=0) + 1e-5)
    return A, B, v_map, target_vars


# indices = torch.tensor(np.array([A.row, A.col]))
# values = torch.tensor(A.data, dtype=torch.float64)
# A__ = torch.sparse_coo_tensor(indices, values, size=A.shape)
# A__.coalesce().values().shape
# A__.coalesce().indices()
# torch.ones(A__.coalesce().values().shape)

# if args.constr_scaling:
# DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
# A = A.tocoo()
# start_time = time.time()

# indices = torch.tensor(np.array([A.row, A.col]))
# data = torch.tensor(A.data, dtype=torch.float64)
# A = torch.sparse_coo_tensor(indices, data, size=A.shape)

# max_values = torch.abs(A.to_dense()).max(dim=1).values
# max_values = torch.maximum(max_values, torch.tensor(B).abs())
# inv_max_values = torch.diag(1 / max_values)

# A = torch.sparse.mm(inv_max_values.to_sparse(), A)
# B = torch.sparse.mm(inv_max_values.to_sparse(), torch.tensor(B).reshape([-1, 1]))

# runtime = time.time() - start_time
