import argparse
import multiprocessing as mp
import os.path
import pickle
from multiprocessing import Process, Queue

import gurobipy as gp
import numpy as np

from helper import get_a_new2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def solve_grb(filepath, log_dir, settings):
    gp.setParam("LogToConsole", 0)
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings["maxsol"]
    m.Params.PoolSearchMode = settings["mode"]
    m.Params.TimeLimit = settings["maxtime"]
    m.Params.Threads = settings["threads"]
    log_path = os.path.join(log_dir, os.path.basename(filepath) + ".log")
    with open(log_path, "w"):
        pass

    m.Params.LogFile = log_path
    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr("SolCount")

    mvars = m.getVars()
    # get variable name,
    oriVarNames = [var.varName for var in mvars]

    # varInds = np.arange(0, len(oriVarNames))

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    sol_data = {
        "var_names": oriVarNames,
        "sols": sols,
        "objs": objs,
    }
    return sol_data


def collect(args, ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            break
        if ins_dir.split("/")[-2] == "CJ" and filename[2] == "L":
            settings["maxtime"] *= 2

        filepath = os.path.join(ins_dir, filename)
        sol_path = os.path.join(sol_dir, filename + ".sol")
        bg_path = os.path.join(bg_dir, filename + ".bg")
        print(f"dataset dir: {filepath}")
        print(f"settings: {settings}")
        if args.genSol and not os.path.exists(sol_path):
            sol_data = solve_grb(filepath, log_dir, settings)
            pickle.dump(sol_data, open(sol_path, "wb"))
        if args.genBG and not os.path.exists(bg_path):
            # get bipartite graph , binary variables' indices
            A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(filepath)
            BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]
            pickle.dump(BG_data, open(bg_path, "wb"))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # datasets = ["IP", "WA", "IS", "CA", "NNV", "CJ"]
    # datasets = ["IP", "WA"]
    os.environ["GRB_LICENSE_FILE"] = "/home/thlee/gurobi.lic"
    datasets = ["WA"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="/mnt/disk1/thlee/MILP/pas")
    parser.add_argument("--nWorkers", type=int, default=80)
    parser.add_argument("--maxTime", type=int, default=3600)
    parser.add_argument("--maxStoredSol", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--genSol", type=str2bool, default=True)
    parser.add_argument("--genBG", type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    # gurobi settings
    SETTINGS = {
        "maxtime": args.maxTime,
        "mode": 2,
        "maxsol": args.maxStoredSol,
        "threads": args.threads,
    }

    for ds in datasets:
        INS_DIR = os.path.join(args.dataDir, f"instance/{ds}/train")
        DS_DIR = os.path.join(args.dataDir, f"dataset/{ds}/train")
        SOL_DIR = f"{DS_DIR}/solution"
        NBP_DIR = f"{DS_DIR}/NBP"
        LOG_DIR = f"{DS_DIR}/logs"
        BG_DIR = f"{DS_DIR}/BG"

        if not os.path.isdir(f"{DS_DIR}"):
            os.makedirs(DS_DIR, exist_ok=True)
        if not os.path.isdir(f"{SOL_DIR}"):
            os.mkdir(f"{SOL_DIR}")
        if not os.path.isdir(f"{NBP_DIR}"):
            os.mkdir(f"{NBP_DIR}")
        if not os.path.isdir(f"{LOG_DIR}"):
            os.mkdir(f"{LOG_DIR}")
        if not os.path.isdir(f"{BG_DIR}"):
            os.mkdir(f"{BG_DIR}")

        filenames = os.listdir(INS_DIR)
        filenames = [file for file in filenames if not file.endswith(".json")]

        q = Queue()
        # add ins
        for filename in filenames:
            # if "D2S01" not in filename:
            #     continue
            # print(filename)
            q.put(filename)
        # add stop signal
        for _ in range(args.nWorkers):
            q.put(None)

        ps = []
        for _ in range(args.nWorkers):
            p = Process(target=collect, args=(args, INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print("done")
