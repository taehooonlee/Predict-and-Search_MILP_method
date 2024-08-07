import os
import pickle
from glob import glob

import gurobipy as gp
import numpy as np

ins_name = "D1S01"
filepath = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/original/{ins_name}.mps.gz"
gp.setParam("LogToConsole", 0)
m = gp.read(filepath)

# p = m.presolve()
# len(p.getVars())


instances = os.listdir("/mnt/disk1/thlee/MILP/pas/instance/CJ/original/")
instances = [ins.split(".")[0] for ins in instances]
for ins_name in instances:
    print(ins_name)
    ins_path = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/original/{ins_name}.mps.gz"
    sol_path = f"/mnt/disk1/thlee/MILP/pas/dataset/CJ/train/solution/{ins_name}.mps.gz.sol"

    m = gp.read(ins_path)
    mvars = m.getVars()
    oriVarNames = {var.varName for var in mvars}

    with open(sol_path, "rb") as f:
        solData = pickle.load(f)
    solVarNames = set(solData["var_names"])

    if len(oriVarNames - solVarNames) != 0:
        print(oriVarNames - solVarNames)
    if len(solVarNames - oriVarNames) != 0:
        print(solVarNames - oriVarNames)


settings = {
    "maxtime": 600,
    "mode": 2,
    "maxsol": 500,
    "threads": 1,
}
for ins_name in instances:
    if ins_name[1:3] != "1S":
        continue
    print(ins_name)

    org_ins_path = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/original/{ins_name}.mps.gz"
    new_ins_path = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/train/{ins_name}.mps.gz"

    m.Params.PoolSolutions = settings["maxsol"]
    m.Params.PoolSearchMode = settings["mode"]
    m.Params.TimeLimit = settings["maxtime"]
    m.Params.Threads = settings["threads"]

    m = gp.read(org_ins_path)
    p = m.presolve()
    p.write(new_ins_path)

b_list, c_list, i_list = [], [], []
for ins_name in instances:
    if ins_name[1:3] != "1S":
        continue
    print(ins_name)

    new_ins_path = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/train/{ins_name}.mps.gz"

    m = gp.read(new_ins_path)
    mvars = m.getVars()

    var_type = [var.VType for var in mvars]
    for t, v in zip(np.unique(var_type, return_counts=True)[0], np.unique(var_type, return_counts=True)[1]):
        if t == "B":
            b_list.append(v)
        elif t == "C":
            c_list.append(v)
        elif t == "I":
            i_list.append(v)
np.mean(b_list)
np.mean(c_list)
np.mean(i_list)

np.mean(b_list) / (np.mean(c_list) + np.mean(i_list)) * 100


# the number of constraints of each dataset
task = "CJ/original"
ins_dir = os.path.join("/mnt/disk1/thlee/MILP/pas/instance", task)
instances = os.listdir(ins_dir)
instances = [ins.split(".")[0] for ins in instances]
len_dict = {}
for comp in ["1", "2", "3"]:
    for size in ["S", "M", "L"]:
        len_dict[f"D{comp}{size}"] = []

for ins_name in instances:
    print(ins_name)
    # ins_path = f"/mnt/disk1/thlee/MILP/pas/instance/CJ/original/{ins_name}.mps.gz"
    ins_path = os.path.join(ins_dir, f"{ins_name}.mps.gz")

    m = gp.read(ins_path)
    len_dict[ins_name[:3]].append(m.numConstrs)

for k, v in len_dict.items():
    print(f"{k}: {np.mean(v)}")


# the number of var/const of other tasks
task = "AN/train"
ins_dir = os.path.join("/mnt/disk1/thlee/MILP/pas/instance", task)
instances = glob(f"{ins_dir}/*")
instances = [i for i in instances if i.endswith(".mps.gz")]


b_list, c_list, i_list = [], [], []
for ins_path in instances:
    m = gp.read(ins_path)
    mvars = m.getVars()

    var_type = [var.VType for var in mvars]
    for t, v in zip(np.unique(var_type, return_counts=True)[0], np.unique(var_type, return_counts=True)[1]):
        print(np.unique(var_type, return_counts=True)[0])
        if t == "B":
            b_list.append(v)
        elif t == "C":
            c_list.append(v)
        elif t == "I":
            i_list.append(v)
np.mean(b_list)
np.mean(c_list)
np.mean(i_list)
