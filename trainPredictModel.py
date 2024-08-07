import argparse
import os
import random
import time
from datetime import datetime

import torch
import torch_geometric
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def EnergyWeightNorm(task):
    if task == "IP":
        return 1
    elif task == "WA":
        return 100
    elif task == "CJ":
        return 10000000
    elif task == "IS":
        return -100
    elif task == "CA":
        return -1000


def train(args, predict, data_loader, optimizer=None, weight_norm=1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    # predict = PredictModel
    # data_loader = train_data
    # data_loader = valid_data

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(data_loader):
            # break
            # for step, batch in enumerate(data_loader):
            batch = batch.to(args.device)
            # print(f"batch print: {batch}")
            # print(f"edge_index print: {batch.edge_index}")
            # print(f"batch index print: {batch.batch}")
            # print(f"ptr print: {batch.ptr}")
            # get target solutions in list format
            solInd = batch.nsols
            target_sols = []
            target_vals = []
            solEndInd = 0
            valEndInd = 0

            for i in range(solInd.shape[0]):  # for in batch
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # remove nan value
            # predict the binary distribution, BD
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            BD = BD.sigmoid()

            # compute loss
            loss = 0
            # calculate weights
            index_arrow = 0
            # print("start calculate loss  :")
            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
                # compute weight
                n_vals = vals
                exp_weight = torch.exp(-n_vals / weight_norm)
                weight = exp_weight / exp_weight.sum()

                # get a binary mask
                varInds = batch.varInds[
                    ind
                ]  # ind가 의미하는 것이 하나의 인스턴스에 있는 50개의 솔루션 중 1개를 말하는 것인지? 50개의 솔루션으로 weight를 구하기 때문에 인스턴스로 보는 것이 맞을듯
                varname_map = varInds[0][0]  # 얘네는 확실히 varname_map이랑 b_vars가 맞음
                b_vars = varInds[1][0].long()

                # get binary variables
                sols = sols[:, varname_map][:, b_vars]

                # cross-entropy
                n_var = batch.ntvars[ind]
                pre_sols = BD[index_arrow : index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var
                pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols == 1).float()
                neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols == 0).float()
                sum_loss = pos_loss + neg_loss

                sample_loss = sum_loss * weight[:, None]
                loss += sample_loss.sum()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    # data hyperparameter
    parser.add_argument("--dir_base", type=str, default="/mnt/disk1/thlee/MILP/pas")
    parser.add_argument("--task_name", type=str, default="IP")
    parser.add_argument("--graph_type", type=str, default="BG")
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--scaling", type=str2bool, default=True)
    # training hyperparameter
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--l_rate", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=9999)
    args = parser.parse_args()
    print(args)

    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")

    dir_graph = os.path.join(args.dir_base, f"dataset/{args.task_name}/train/{args.graph_type}")
    dir_sol = os.path.join(args.dir_base, f"dataset/{args.task_name}/train/solution")

    # set folder
    curr_time = datetime.now().strftime("%y%m%d-%H%M%S")
    train_task = f"{args.task_name}_train_{args.graph_type}"
    model_save_path = os.path.join(args.dir_base, f"pretrain/{train_task}/{curr_time}")
    # log_save_path = os.path.join(args.dir_base, f"train_logs/{train_task}")
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    # if not os.path.isdir(log_save_path):
    #     os.makedirs(log_save_path, exist_ok=True)
    log_file = open(f"{model_save_path}/train_result.log", "wb")
    st = f"@args:\t{args}\n"
    log_file.write(st.encode())
    log_file.flush()

    # set files for training
    graph_type = args.graph_type.lower()
    sample_names = os.listdir(dir_graph)
    sample_files = [(os.path.join(dir_graph, name), os.path.join(dir_sol, name).replace(graph_type, "sol")) for name in sample_names]
    random.seed(0)
    random.shuffle(sample_files)
    train_files = sample_files[: int(0.80 * len(sample_files))]
    valid_files = sample_files[int(0.80 * len(sample_files)) :]
    # print(valid_files)
    # exit()
    if graph_type == "bg":
        if args.task_name == "IP":
            # Add position embedding for IP model, due to the strong symmetry
            from GCN import GNNPolicy_position as GNNPolicy
            from GCN import GraphDataset_position as GraphDataset
        else:
            from GCN import GNNPolicy, GraphDataset
    elif graph_type == "hg":
        from GCN import GNNPolicy
        from GCN import GraphDataset_HG as GraphDataset

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("dataset has been loaded")

    PredictModel = GNNPolicy().to(args.device)
    print("model has been loaded")

    optimizer = torch.optim.Adam(PredictModel.parameters(), lr=args.l_rate)

    weight_norm = EnergyWeightNorm(args.task_name)
    best_val_loss = 99999
    for epoch in range(args.n_epochs):
        begin = time.time()
        train_loss = train(args, PredictModel, train_loader, optimizer, weight_norm)
        print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
        if epoch % 10 == 0:
            valid_loss = train(args, PredictModel, valid_loader, None, weight_norm)
            print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(PredictModel.state_dict(), os.path.join(model_save_path, "model_best.pth"))
            torch.save(PredictModel.state_dict(), os.path.join(model_save_path, "model_last.pth"))
        else:
            valid_loss = "-"
        st = f"@epoch:{epoch}\tTrain_loss:{train_loss}\tValid_loss:{valid_loss}\tTIME:{time.time()-begin}\n"
        log_file.write(st.encode())
        log_file.flush()
    print("done")
