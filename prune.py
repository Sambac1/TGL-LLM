import os
import sys
import pickle
import yaml
import argparse
import logging
import time
import numpy as np
from collections import Counter
import json
from torch.utils.tensorboard import SummaryWriter

import dgl
import torch
from tqdm import tqdm
import random
import modules.utils_pretrain as utils
from modules.regcn import REGCN
from torch.autograd import grad
import math

def get_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu", type=str, default=0, help="which gpu to use")
    parser.add_argument("-d", "--dataset", type=str, default="ICEWS14s", help="which dataset to use, options: ICEWS14, ICEWS14s, ICEWS05-15, ICEWS18, GDELT")
    parser.add_argument("-m", "--model", type=str, default="REGCN", help="which model to use, options: REGCN")
    parser.add_argument("-i", "--info", type=str, default="", help="addtional info for certain run")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--hist_len", type=int, default=3, help="hist len")
    # configuration for contrasive loss
    args = parser.parse_args()

    return args


def hvp(y, w, v):
    """
    ``y`` is the scalor of loss value
    ``w`` is the model parameters
    ``v`` is the H^{-1}v at last step
    """
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    # print(f'first_grads:{first_grads}')
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)
    # print(f'elemwise_products:{elemwise_products}')

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)
    # print(f'return_grads:{return_grads}')

    return return_grads

def get_sample_loss(model,sample_data,hist_len,train_times,graph_dict,device):
    train_sample_num=sample_data[3]
    train_sample_num_idx = train_times.index(train_sample_num)
    if train_sample_num_idx - hist_len < 0:
        hist_list = train_times[0:train_sample_num_idx]
    else:
        hist_list = train_times[train_sample_num_idx - hist_len: train_sample_num_idx]

    g_list = [graph_dict[tim].to(device) for tim in hist_list]
    if len(hist_list) ==0:
        print(hist_list)
    output = torch.LongTensor(np.delete(sample_data,3)).reshape(1,3).to(device)

    loss_ent = model(g_list, output)
    # print(loss_ent)
    return loss_ent

def estimate_hv(train_data, model, h_estimate, v,hist_len,train_times,graph_dict,device):

    recursion_depth = 5000
    damp = 0.01
    scale = 25
    
    for _ in tqdm(range(recursion_depth),total=recursion_depth):

        random_idx = random.choice(list(range(len(train_data))))
        if train_data[random_idx,3] == 0: 
            continue
        loss = get_sample_loss(model, train_data[random_idx,:],hist_len,train_times,graph_dict,device)

        params = [ p for n, p in model.named_parameters() if n == "decoder.fc.weight" ] # we follow previous work to calculate the last linear layer for high efficiency

        hv = hvp(loss, params, h_estimate)

        # Recursively caclulate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            h_estimate = [_.detach() for _ in h_estimate]

    hv = [_.reshape(-1) for _ in hv]
    h_estimate = [_.reshape(-1) for _ in h_estimate]
    v = [_.reshape(-1) for _ in v]

    return h_estimate


def main():
    conf = yaml.safe_load(open("./config_pretrain.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    conf = conf[dataset_name]
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["model"] = paras["model"]
    conf["dataset"] = dataset_name
    conf["data_path"] = conf["path"] + "/" + conf["dataset"]
    conf["lr"] = paras['lr']
    conf["wd"] = paras['wd']
    conf['hist_len'] = paras['hist_len']

    os.environ['CUDA_VISIBLE_DEVICES'] = conf['gpu']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    # set influence score parameters
    hard_prune = 0.3
    k=25
    n_fewshot=100000

    # load data
    print("loading sparsity group")

    print("loanding training graphs...")
    with open(os.path.join(conf["data_path"], 'graph_dict.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)


    data = utils.RGCNLinkDataset(conf["dataset"], conf["path"])
    data.load()
    train_data = data.train
    train_list = utils.split_by_time(train_data)
    train_times = np.array(sorted(set(train_data[:, 3]))) 
    train_times=list(train_times)

    num_ents = data.num_nodes
    num_rels = data.num_rels
    conf["num_ent"] = num_ents
    conf["num_rel"] = num_rels
    
    # initialize log
    model_name = "{}-{}-{}-lr{}-wd{}-dim{}-histlen{}-layers{}".format(conf["model"], conf["info"], conf["decoder_name"], conf["lr"], conf["wd"], conf["h_dim"], conf['hist_len'],conf['n_layers'])
    model_path = './checkpoints/regcn/{}/'.format(conf["dataset"])
    model_state_file = model_path + model_name

    # build model
    model = REGCN(conf)
    model.to(device)

    checkpoint = torch.load(model_state_file, map_location=conf["device"])
    logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
    logging.info("\n"+"-"*10+"start testing"+"-"*10+"\n")
    model.load_state_dict(checkpoint['state_dict'])

    device = conf['device']
    
    model.eval()

    idx = [_ for _ in range(len(train_list))]
    random.shuffle(idx)
    
    losses = []
    hist_len = conf["hist_len"]

    # # step 1. calculate constant vector in Eq.(11)
    v_list = []
    for batch_idx, train_sample_num in enumerate(tqdm(idx)):
        if train_sample_num == 0: 
            continue

        if train_sample_num - hist_len < 0:
            hist_list = train_times[0:train_sample_num]
        else:
            hist_list = train_times[train_sample_num - hist_len: train_sample_num]

        g_list = [graph_dict[tim].to(device) for tim in hist_list]
        output = torch.LongTensor(train_list[train_sample_num]).to(device)
        loss_ent = model(g_list, output)
        loss = loss_ent
        for name, params in model.named_parameters():
            if name == 'decoder.fc.weight':
                v_list.append(list(grad(loss, params))[0].unsqueeze(0))
    v = [torch.mean(torch.cat(v_list, dim=0), dim=0)]
    h_estimate = v.copy()

    H_inverse = estimate_hv(train_data,model,h_estimate, v,hist_len,train_times,graph_dict,device)
    H_inverse = [ _.data for _ in H_inverse]

    # step 3. calculate the influence score for each sample. <- Eq.(11)
    influence_score = [0 for _ in range(len(train_data))]
    for idx in tqdm(range(len(train_data)), total=len(train_data)):
        # sample = cal_grad_z(idx, trainer)
        if train_data[idx,3] == 0: 
            continue
        loss = get_sample_loss(model, train_data[idx,:],hist_len,train_times,graph_dict,device)
        for name, params in model.named_parameters():
            if name == "decoder.fc.weight":
                sample = list(grad(loss, params))

        score = torch.matmul(H_inverse[0], sample[0].view(-1).T)
        influence_score[idx] = score

    influence_score = torch.tensor(influence_score)

    scores_sorted, indices = torch.sort(influence_score, descending=True)

    n_prune = math.floor(hard_prune * len(scores_sorted))
    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    print(f"** after hard prune with {hard_prune*100}% data:", len(scores_sorted))

    # split scores into k ranges
    s_max = torch.max(scores_sorted)
    s_min = torch.min(scores_sorted)
    print("== max socre:", s_max)
    print("== min score:", s_min)
    interval = (s_max - s_min) / k

    s_split = [min(s_min + (interval * _), s_max)for _ in range(1, k+1)]

    score_split = [[] for _ in range(k)]
    for idxx, s in enumerate(scores_sorted):
        for idx, ref in enumerate(s_split):
            if s.item() <= ref:
                score_split[idx].append({indices[idxx].item():s.item()})
                break

    coreset = []
    m = n_fewshot
    while len(score_split):
        # select the group with fewest samples
        group = sorted(score_split, key=lambda x:len(x))
        if len(group) > 3:
            print("** sorted group length:", len(group[0]), len(group[1]), len(group[2]), len(group[3]),"...")
        
        group = [strat for strat in group if len(strat)]
        if len(group) > 3:
            print("** sorted group length after removing empty ones:", len(group[0]), len(group[1]), len(group[2]), len(group[3]),"...")

        budget = min(len(group[0]), math.floor(m/len(group)))
        print("** budget for current fewest group:", budget)
        
        # random select and add to the fewshot indices list
        fewest = group[0]
        selected_idx = random.sample([list(_.keys())[0] for _ in fewest], budget)
        coreset.extend(selected_idx)

        # remove the fewest group
        score_split = group[1:]
        m = m - len(selected_idx)
    
    save_path = './data/{}/'.format(conf["dataset"])
    save_path = save_path + 'sample_10w_coreset_{}.json'.format(conf["dataset"])
    with open(save_path,'w') as f:
        json.dump(coreset,f,ensure_ascii=False,indent=2)

if __name__ == '__main__':
    main()
