import argparse
import multiprocessing as mp
import os.path
import pickle
from multiprocessing import Process, Queue

import gurobipy as gp
import numpy as np
import pyscipopt as scp
from pyscipopt import SCIP_PARAMSETTING

from helper import ds2type, get_a_new2, get_HG_from_GRB


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def solve_scip(filepath, log_dir, settings):
    m1 = scp.Model()
    # m1.setParam("heuristics/active", False)
    # m1.setParam("emphasis/heuristics", "off")
    # m1.setParam("emphasis/branching", True)
    m1.setParam("presolving/maxrounds", 0)
    m1.setParam("separating/maxrounds", 0)
    m1.setParam("separating/maxcuts", 0)

    m1.setParam("limits/time", settings["maxtime"])
    m1.setParam("limits/solutions", settings["maxsol"])
    m1.hideOutput(True)
    m1.setParam("randomization/randomseedshift", 0)
    m1.setParam("randomization/lpseed", 0)
    m1.setParam("randomization/permutationseed", 0)
    m1.setHeuristics(SCIP_PARAMSETTING.DEFAULT)  # MIP focus
    log_path = os.path.join(log_dir, os.path.basename(filepath) + ".log")
    with open(log_path, "w"):
        pass
    m1.setLogfile(log_path)
    m1.readProblem(filepath)
    m1.optimize()


def solve_grb(filepath, log_dir, settings):
    gp.setParam("LogToConsole", 0)
    m = gp.read(filepath)
    m.setParam("Heuristics", 0)
    m.setParam("Presolve", 0)
    m.setParam("Cuts", 0)

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


def collect(args, ins_dir, q, sol_dir, log_dir, bg_dir, hg_dir, settings):
    # ins_dir = INS_DIR
    # log_dir = LOG_DIR
    # settings = SETTINGS
    # filename = filenames[0]
    while True:
        filename = q.get()
        if not filename:
            break
        if ins_dir.split("/")[-2] == "CJ" and filename[2] == "L":
            settings["maxtime"] *= 2

        filepath = os.path.join(ins_dir, filename)
        sol_path = os.path.join(sol_dir, filename + ".sol")
        bg_path = os.path.join(bg_dir, filename + ".bg")
        hg_path = os.path.join(hg_dir, filename + ".hg")
        print(f"dataset dir: {filepath}")
        print(f"settings: {settings}")

        if args.genSol and not os.path.exists(sol_path):
            if args.solver == "gurobi":
                sol_data = solve_grb(filepath, log_dir, settings)
                pickle.dump(sol_data, open(sol_path, "wb"))
            elif args.solver == "scip":
                solve_scip(filepath, log_dir, settings)
        if args.genBG and not os.path.exists(bg_path):
            # get bipartite graph , binary variables' indices
            A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(args, filepath)
            BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]
            pickle.dump(BG_data, open(bg_path, "wb"))
        if args.genHG and not os.path.exists(hg_path):
            A, B, v_map, target_vars, sense, coef_obj = get_HG_from_GRB(args, filepath)
            HG_data = [A, B, v_map, target_vars, sense, coef_obj]
            pickle.dump(HG_data, open(hg_path, "wb"))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # datasets = ["IP", "WA", "IS", "CA", "NNV", "CJ"]
    # datasets = ["IP", "WA"]
    os.environ["GRB_LICENSE_FILE"] = "/home/thlee/gurobi.lic"
    # datasets = ["IP/train"]
    # datasets = ["CJ-S/train"]
    # datasets = ["IP/train", "CJ/train"]
    # datasets = ["WA/train", "WA/test"]
    # datasets = ["18-9/test", "30-20/test", "50-40/test", "50-100/test"]
    # datasets = ["50-100/test/same_241024-175706", "50-100/test/same_241024-175823"]
    # datasets = ["50-100/test/same_241025-005320", "50-100/test/same_241025-005436"]
    # datasets = ["50-100/test/same_241025-114447", "50-100/test/same_241025-114548"]
    # datasets = ["500-1000/test/same_241025-222716", "500-1000/test/same_241025-224146"]
    # datasets = ["18-9/test/diff", "150-500/test/diff", "250-1000/test/diff"]
    # datasets = ["100-300/test/diff"]
    datasets = ["20-50/test/diff"]
    # datasets = ["50-100/test/diff"]
    # datasets = ["50-100/test/diff", "150-1000/test/diff"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="/mnt/disk1/thlee/MILP/pas")
    parser.add_argument("--nWorkers", type=int, default=25)
    # parser.add_argument("--maxTime", type=int, default=3600)
    parser.add_argument("--maxTime", type=int, default=600)
    # parser.add_argument("--maxTime", type=int, default=600)
    parser.add_argument("--maxStoredSol", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--genSol", type=str2bool, default=True)
    parser.add_argument("--genBG", type=str2bool, default=False)
    parser.add_argument("--genHG", type=str2bool, default=False)

    parser.add_argument("--solver", type=str, default="gurobi")
    # parser.add_argument("--solver", type=str, default="scip")
    # parser.add_argument("--scaling", type=str2bool, default=True)
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
        if ds.split("/")[0] in ds2type.keys():
            args.var_type = ds2type[ds.split("/")[0]]
        else:
            args.var_type = "I"
        print(f"args.var_type: {args.var_type}")
        INS_DIR = os.path.join(args.dataDir, f"instance/{ds}")
        DS_DIR = os.path.join(args.dataDir, f"dataset/{ds}")
        NBP_DIR = f"{DS_DIR}/NBP"
        # LOG_DIR = f"{DS_DIR}/logs"
        BG_DIR = f"{DS_DIR}/BG"
        HG_DIR = f"{DS_DIR}/HG"
        if args.solver == "gurobi":
            LOG_DIR = f"{DS_DIR}/logs"
            SOL_DIR = f"{DS_DIR}/solution"
        elif args.solver == "scip":
            LOG_DIR = f"{DS_DIR}/logs_scip"
            SOL_DIR = f"{DS_DIR}/solution_scip"

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
        if not os.path.isdir(f"{HG_DIR}"):
            os.mkdir(f"{HG_DIR}")

        filenames = os.listdir(INS_DIR)
        filenames = sorted([file for file in filenames if not file.endswith(".json")])

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
            p = Process(target=collect, args=(args, INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, HG_DIR, SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print("done")
