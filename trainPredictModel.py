import os
import random
import sys
import time
from datetime import datetime
from os.path import join as pjoin

import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch_sparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from evidential_dl import evidential_classification, evidential_regression
from helper import create_parser, ds2type, shrink_modulator


def EnergyWeightNorm(task):
    if task == "IP":
        return 1
    elif task == "WA":
        return 100
    elif task == "CJ":
        return 1000000000
    elif task == "IS":
        return -100
    elif task == "CA":
        return -1000


def train2(args, model, data_loader, optimizer=None, weight_norm=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    # model = PredictModel
    # data_loader = train_loader
    # data_loader = valid_loader
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss, mean_loss_mae, mean_loss_int = 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(data_loader):
            # break
            batch = batch.to(args.device)

            n_sols = batch.nsols
            loss, loss_mae, loss_int = 0, 0, 0  # compute loss
            idx_sol_end = 0
            target_sols, n_vars = [], []
            for i in range(batch.num_graphs):  # for-loop in batch, i is index for each instance graph in batch
                n_var = len(batch.varInds[i][0][0])
                idx_sol_start = idx_sol_end
                idx_sol_end = n_sols[i] * n_var + idx_sol_start
                sols = batch.solutions[idx_sol_start:idx_sol_end].reshape(-1, n_var)
                target_sols.append(sols)
                n_vars.append(n_var)

            if args.graph_type == "BG":
                # predict the binary distribution
                batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # remove nan value
                output = model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            elif args.graph_type == "HG":
                output = model(batch.edge_index, batch.coef, batch.rhs)

            idx_arrow = 0
            for idx_ins, (sols, n_var) in enumerate(zip(target_sols, n_vars)):
                # break
                # get target variables
                varInds = batch.varInds[idx_ins]
                varname_map = varInds[0][0]
                target_vars = varInds[1][0]

                sols = torch.abs(sols[:, varname_map][:, target_vars])
                pre_sols = output[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]

                sum_loss_mae = F.l1_loss(torch.broadcast_to(pre_sols, sols.shape), sols, reduction="none")  # MAE
                loss_mae += sum_loss_mae.mean()
                if ds2type[args.task_name] == "B":
                    # loss_int += torch.min(pre_sols, (1 - pre_sols)).mean()
                    loss_int += -torch.log(pre_sols).mean()
                elif ds2type[args.task_name] == "I":
                    sum_loss_int = (pre_sols < 0) * F.huber_loss(pre_sols, torch.zeros_like(pre_sols), reduction="none")
                    loss_int += sum_loss_int.mean()

                idx_arrow = idx_arrow + n_var

            num_edges = int(batch.edge_index[1].max()) + 1
            edge_index, coef = torch_sparse.coalesce(batch.edge_index.flip([0]), batch.coef, num_edges, batch.num_nodes, op="mean")
            lhs = torch_sparse.spmm(edge_index, coef, num_edges, batch.num_nodes, output)

            loss = F.huber_loss(lhs, batch.rhs[:, [1]], reduction="none")
            loss_eq = loss * (batch.rhs[:, [0]] == 0)
            loss_le = loss * (batch.rhs[:, [0]] == 1) * (lhs > batch.rhs[:, [1]])
            loss_ge = loss * (batch.rhs[:, [0]] == 2) * (lhs < batch.rhs[:, [1]])
            loss_const = (loss_eq + loss_le + loss_ge).mean()

            loss_int /= batch.num_graphs

            loss_obj = torch.matmul(batch.coef_obj, output) / batch.num_graphs
            loss_obj = torch.abs(torch.log(loss_obj + 1e-8))
            # loss_obj = torch.max(torch.log(loss_obj + 1e-8), torch.zeros_like(loss_obj))
            # loss_obj = -torch.min(torch.log(loss_obj + 1e-8), torch.zeros_like(loss_obj))

            loss = loss_const + loss_int + args.lamb_obj * loss_obj

            if loss < 0.05:
                loss = torch.matmul(batch.coef_obj, output) / batch.num_graphs

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item()
            mean_loss_int += loss_int.item()
            mean_loss_mae += loss_mae.item() / batch.num_graphs
    mean_loss /= len(data_loader)
    mean_loss_int /= len(data_loader)
    mean_loss_mae /= len(data_loader)
    print(f"MAE: {mean_loss_mae:0.4f}")
    print(f"mean_loss_int: {mean_loss_int:0.4f}")
    obj_val = torch.matmul(batch.coef_obj.flatten(), output) / batch.num_graphs
    print(f"OBJ value: {obj_val.item():0.4f}")
    if optimizer is None:
        pred_y = pre_sols.cpu()
        sols = sols.cpu().T
        df_pred_test = pd.DataFrame(sols)
        cols = df_pred_test.columns
        df_pred_test["pred_y"] = pred_y
        df_pred_test = df_pred_test[["pred_y"] + list(cols)]
        df_pred_test.sort_values(by=[0, "pred_y"], axis=0, ascending=False, inplace=True)
        df_pred_test.to_csv(f"{model_save_path}/pred_results.csv", index=False)

    return mean_loss, mean_loss_mae


def train(args, model, data_loader, optimizer=None, weight_norm=1, eps=1e-5):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    # model = PredictModel
    # data_loader = train_loader
    # data_loader = valid_loader
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss, mean_loss_mae, num_labels, n_samples_processed = 0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(data_loader):
            # break
            batch = batch.to(args.device)
            n_sols = batch.nsols
            loss, loss_mae = 0, 0  # compute loss
            idx_sol_end, idx_val_end = 0, 0
            target_sols, target_vals, n_vars = [], [], []

            for i in range(batch.num_graphs):  # for-loop in batch, i is index for each instance graph in batch
                n_var = len(batch.varInds[i][0][0])
                idx_sol_start = idx_sol_end
                idx_sol_end = n_sols[i] * n_var + idx_sol_start
                idx_val_start = idx_val_end
                idx_val_end = idx_val_end + n_sols[i]
                sols = batch.solutions[idx_sol_start:idx_sol_end].reshape(-1, n_var)
                vals = batch.objVals[idx_val_start:idx_val_end]

                target_sols.append(sols)
                target_vals.append(vals)
                n_vars.append(n_var)

            if args.graph_type == "BG":
                # predict the binary distribution
                batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # remove nan value
                output = model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            elif args.graph_type == "HG":
                output = model(batch.edge_index, batch.coef, batch.rhs)
                # emb_0 = torch.zeros((model.x_const.shape[0], 1), device=model.x_const.device)
                # loss_rhs = F.mse_loss(model.x_const, model.rhs_emb(emb_0), reduction="mean")
                # loss += loss_rhs
                # print(f"Loss of RHS: {loss_rhs}")
                # print(f"mean of const: {model.x_const.mean()}")
                # print(f"std of const: {model.x_const.std()}")
                # print(f"max of const: {model.x_const.max()}")
                # print(f"min of const: {model.x_const.min()}")
                # print(f"max of rhs_emb: {model.rhs_emb(emb_0).max()}")
                # print(f"min of rhs_emb: {model.rhs_emb(emb_0).min()}")

            idx_arrow = 0
            for idx_ins, (sols, vals, n_var) in enumerate(zip(target_sols, target_vals, n_vars)):
                # break
                # compute weight
                weight = (vals - vals.min() + eps) / (vals.max() - vals.min() + 2 * eps)
                weight = -torch.log(weight)
                weight = weight / weight.sum()

                # exp_weight = torch.exp(-vals / weight_norm)
                # weight = exp_weight / exp_weight.sum()

                # get target variables
                varInds = batch.varInds[idx_ins]
                varname_map = varInds[0][0]
                target_vars = varInds[1][0]
                # print(f"target_vars: {target_vars.shape}")
                sols = torch.abs(sols[:, varname_map][:, target_vars])

                # cross-entropy
                if not args.evi_loss:
                    pre_sols = output[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]
                    if ds2type[args.task_name] == "B":
                        sum_loss = F.binary_cross_entropy(torch.broadcast_to(pre_sols, sols.shape), sols, reduction="none")
                        # pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols == 1).float()
                        # neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols == 0).float()
                        # sum_loss = pos_loss + neg_loss

                        # weight_disc = torch.min(pre_sols, (1 - pre_sols)).clone().detach()
                        # weight_disc = torch.exp(weight_disc)
                        # loss_disc = torch.min(output, (1 - output)).mean()
                        # loss += loss_disc

                    elif ds2type[args.task_name] == "I":
                        sum_loss = F.huber_loss(torch.broadcast_to(pre_sols, sols.shape), sols, reduction="none")  # HUBER
                        # sum_loss = F.mse_loss(torch.broadcast_to(pre_sols, sols.shape), sols, reduction="none")  # MSE
                    sum_loss_mae = F.l1_loss(torch.broadcast_to(pre_sols, sols.shape), sols, reduction="none")  # MAE

                else:
                    if ds2type[args.task_name] == "I":
                        mu, v, alpha, beta = output
                        mu_ = mu[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]
                        v_ = v[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]
                        alpha_ = alpha[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]
                        beta_ = beta[idx_arrow : idx_arrow + n_var].squeeze()[target_vars]

                        mu_ = torch.broadcast_to(mu_, sols.shape)
                        v_ = torch.broadcast_to(v_, sols.shape)
                        alpha_ = torch.broadcast_to(alpha_, sols.shape)
                        beta_ = torch.broadcast_to(beta_, sols.shape)

                        # sum_loss = evidential_regression((mu_, v_, alpha_, beta_), sols, lamb=min(1e-2, epoch / args.lamb_edl))
                        sum_loss = evidential_regression((mu_, v_, alpha_, beta_), sols, lamb=args.lamb_edl)
                    sum_loss_mae = F.l1_loss(mu_, sols, reduction="none")  # MAE

                if args.loss_penalty == "focal":
                    # pos_loss = (sols == 1) * sum_loss
                    pos_loss = (sols == 1) * ((1 - pre_sols) ** args.focal_gamma) * sum_loss
                    neg_loss = (sols == 0) * ((pre_sols) ** args.focal_gamma) * sum_loss
                    sum_loss = pos_loss + neg_loss
                elif args.loss_penalty == "shrink":
                    sum_loss *= shrink_modulator(sum_loss)
                # elif ds2type[args.task_name] == "B" and args.evi_loss:
                #     loss = evidential_classification(pred, y, lamb=1e-2)

                idx_arrow = idx_arrow + n_var
                sample_loss = sum_loss * weight[:, None] * sols.shape[0]
                # sample_loss = sum_loss * weight[:, None] * sols.shape[0]
                # loss += sample_loss.sum()
                loss += sample_loss.sum(axis=0).mean()

                sample_loss_mae = sum_loss_mae * weight[:, None]
                # loss_mae += sample_loss_mae.sum()
                loss_mae += sample_loss_mae.sum(axis=0).mean()

                # num_labels += sols.shape[1]
                # num_labels += sols.shape[0] * sols.shape[1]
            loss /= batch.num_graphs
            loss_mae /= batch.num_graphs
            # loss += loss_mae
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()

                # for name, param in model.rhs_emb.named_parameters():
                #     if param.grad is not None:
                #         print(f"max gradient of {name}:", param.grad.max())
                #         print(f"min gradient of {name}:", param.grad.min())

                optimizer.step()
            mean_loss += loss.item()
            mean_loss_mae += loss_mae.item()
            # n_samples_processed += batch.num_graphs
    # mean_loss /= n_samples_processed
    mean_loss /= len(data_loader)
    mean_loss_mae /= len(data_loader)
    # mean_loss /= num_labels
    # mean_loss_mae /= num_labels

    print(f"MAE: {mean_loss_mae:0.4f}")
    if args.evi_loss and ds2type[args.task_name] == "I":
        print(f"sample v mean, max, min: {v_.mean()}, {v_.max()}, {v_.min()}")
        print(f"sample alpha mean, max, min: {alpha_.mean()}, {alpha_.max()}, {alpha_.min()}")

    if optimizer is None:
        pred_y = pre_sols.cpu()
        sols = sols.cpu().T
        df_pred_test = pd.DataFrame(sols)
        df_pred_test["pred_y"] = pred_y
        df_pred_test = df_pred_test[["pred_y"] + list(range(sols.shape[1]))]
        df_pred_test.sort_values(by=[0, "pred_y"], axis=0, ascending=False, inplace=True)
        df_pred_test.to_csv(f"{model_save_path}/pred_results.csv", index=False)

    return mean_loss, mean_loss_mae


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")

    # create parser
    args = create_parser()
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")

    # args.batch_size = 128
    # args.task_name = "CJ-S"
    # args.graph_type = "HG"
    # args.evi_loss = True
    # args.evi_loss = False
    # args.n_sols = 1
    # args.n_conv = 5

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

    # set folder
    curr_time = datetime.now().strftime("%y%m%d-%H%M%S")
    train_task = f"{args.task_name}_train_{args.graph_type}"
    model_save_path = pjoin(args.dir_base, f"pretrain/{train_task}/{curr_time}")

    # report experiment settings
    exp_info_dir = os.path.join(os.getcwd(), "exp_info")
    if not os.path.isdir(exp_info_dir):
        os.makedirs(exp_info_dir, exist_ok=True)
    f = open(f"{exp_info_dir}/{args.exp_file_name}.txt", "a")
    f.write(f"{model_save_path}\n")
    f.write(f"{args}\n\n")
    f.close()

    # set log file
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    log_file = open(f"{model_save_path}/train_result.log", "wb")
    st = f"@args:\t{args}\n"
    log_file.write(st.encode())
    log_file.flush()

    # set files for training
    dir_graph = pjoin(args.dir_base, f"dataset/{args.task_name}/train/{args.graph_type}")
    dir_sol = pjoin(args.dir_base, f"dataset/{args.task_name}/train/solution")
    # sample_names = os.listdir(dir_graph)
    # sample_files = [(pjoin(dir_graph, name), pjoin(dir_sol, name).replace(args.graph_type.lower(), "sol")) for name in sample_names]
    sample_names = os.listdir(dir_sol)
    sample_files = [(pjoin(dir_graph, name).replace("sol", args.graph_type.lower()), pjoin(dir_sol, name)) for name in sample_names]
    random.shuffle(sample_files)
    train_files = sample_files[: int(0.80 * len(sample_files))]
    valid_files = sample_files[int(0.80 * len(sample_files)) :]

    # load data
    train_data = GraphDataset(args, train_files)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    valid_data = GraphDataset(args, valid_files)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("dataset has been loaded")

    # for i in range(train_data.len()):
    #     try:
    #         train_data.get(i).num_nodes
    #     except:
    #         print(train_data.sample_files[i])
    #         # os.remove(train_data.sample_files[i][0])
    #         # os.remove(train_data.sample_files[i][1])
    #         # os.remove(train_data.sample_files[i][1].replace("/solution/", "/logs/").replace(".sol", ".log"))
    #         # os.remove(train_data.sample_files[i][1].replace("/dataset/", "/instance/").replace(".sol", "").replace("/solution/", "/"))
    # for i in range(valid_data.len()):
    #     try:
    #         valid_data.get(i).num_nodes
    #     except:
    #         print(valid_data.sample_files[i])
    #         # os.remove(valid_data.sample_files[i][0])
    #         # os.remove(valid_data.sample_files[i][1])
    #         # os.remove(valid_data.sample_files[i][1].replace("/solution/", "/logs/").replace(".sol", ".log"))
    #         # os.remove(train_data.sample_files[i][1].replace("/dataset/", "/instance/").replace(".sol", "").replace("/solution/", "/"))
    # sys.exit()
    # list_num_nodes = [train_data.get(i).num_nodes for i in range(train_data.len())] + [valid_data.get(i).num_nodes for i in range(valid_data.len())]
    # if len(set(list_num_nodes)) == 1:
    #     args.num_vars = list_num_nodes[0]
    # else:
    #     sys.exit()
    args.num_vars = train_data.get(0).num_nodes
    # args.n_rhs_features = train_data.get(0).rhs.shape[1]
    print(args.num_vars)

    # load model
    PredictModel = GNNPolicy(args).to(args.device)
    optimizer = torch.optim.Adam(PredictModel.parameters(), lr=args.l_rate, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(PredictModel.parameters(), lr=args.l_rate)
    print(PredictModel)
    print("model has been loaded")
    print(f"Model save path: {model_save_path}")

    # train model
    weight_norm = EnergyWeightNorm(args.task_name)
    best_val_loss = 99999
    for epoch in range(args.n_epochs):
        begin = time.time()
        train_loss, _ = train2(args, PredictModel, train_loader, optimizer, weight_norm)
        print(f"Epoch {epoch} Train loss: {train_loss:0.4f}")
        if epoch % 10 == 0:
            valid_loss, valid_loss_mae = train2(args, PredictModel, valid_loader, None, weight_norm)
            print(f"Epoch {epoch} Valid loss: {valid_loss:0.4f}")
            if valid_loss_mae < best_val_loss:
                best_val_loss = valid_loss_mae
                torch.save(PredictModel.state_dict(), pjoin(model_save_path, "model_best.pth"))
            torch.save(PredictModel.state_dict(), pjoin(model_save_path, "model_last.pth"))
        else:
            valid_loss = "-"
        st = f"@epoch:{epoch}\tTrain_loss:{train_loss}\tValid_loss:{valid_loss}\tTIME:{time.time()-begin}\n"
        log_file.write(st.encode())
        log_file.flush()
    print(f"Model save path: {model_save_path}")
