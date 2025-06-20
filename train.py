import os
import sys
import pickle
import yaml
import argparse
import logging
import time
import numpy as np
from collections import Counter

from torch.utils.tensorboard import SummaryWriter

import dgl
import torch
from tqdm import tqdm
import random
import modules.utils_pretrain as utils
from modules.regcn import REGCN

def get_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu", type=str, default=0, help="which gpu to use")
    parser.add_argument("-d", "--dataset", type=str, default="ICEWS14s", help="which dataset to use, options: ICEWS14, ICEWS14s, ICEWS05-15, ICEWS18, GDELT")
    parser.add_argument("-m", "--model", type=str, default="REGCN", help="which model to use, options: REGCN")
    parser.add_argument("-i", "--info", type=str, default="", help="addtional info for certain run")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--hist_len", type=int, default=3, help="hist len")
    args = parser.parse_args()

    return args


def test_sparsity(conf, model, model_name, history_times, query_times, graph_dict, test_list, all_ans_list, mode='eval'):

    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=conf["device"])
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        logging.info("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    device = conf['device']

    ranks_raw, mrr_raw_list = [], []
    ranks_filter, mrr_filter_list = [], []
    ranks_raw_obj, ranks_raw_sub, ranks_filter_obj, ranks_filter_sub = [], [], [], []

    model.eval()
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        query_time = query_times[time_idx]
        query_idx = np.where(history_times == query_time)[0].item()
        input_list = history_times[query_idx - conf["hist_len"] : query_idx]

        g_list = [graph_dict[tim].to(device) for tim in input_list]

        test_triples_input = torch.LongTensor(test_snap).to(device)

        final_score, test_triples, _, _,_ = model.predict_p(g_list, test_triples_input)

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank_sparse(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000)

        num_triples = len(test_triples)

        ranks_raw_obj.append(rank_raw[:num_triples//2])
        ranks_raw_sub.append(rank_raw[num_triples//2:])
        ranks_filter_obj.append(rank_filter[:num_triples//2])
        ranks_filter_sub.append(rank_filter[num_triples//2:])
        
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
    
    mrr_raw,hits_raw = utils.cal_ranks(ranks_raw, conf["hit_ks"])
    mrr_filter,hits_filter = utils.cal_ranks(ranks_filter, conf["hit_ks"])

    results = {"raw": [mrr_raw, hits_raw], "filter": [mrr_filter, hits_filter]}

    return results


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
    print(device)
    # load data
    print("loading sparsity group")
    sparsity_split_ent, sparsity_split_rel = None, None

    print("loanding training graphs...")
    with open(os.path.join(conf["data_path"], 'graph_dict.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)


    data = utils.RGCNLinkDataset(conf["dataset"], conf["path"])
    data.load()
    train_data = data.train
    train_list = utils.split_by_time(train_data)
    train_times = np.array(sorted(set(train_data[:, 3]))) 
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    
    history_times = np.concatenate((train_times, val_times, test_times), axis=None)

    num_ents = data.num_nodes
    num_rels = data.num_rels
    conf["num_ent"] = num_ents
    conf["num_rel"] = num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_ents, False)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_ents, False)
    
    # initialize log
    model_name = "{}-{}-{}-lr{}-wd{}-dim{}-histlen{}-layers{}".format(conf["model"], conf["info"], conf["decoder_name"], conf["lr"], conf["wd"], conf["h_dim"], conf['hist_len'],conf['n_layers'])
    model_path = './checkpoints/regcn/{}/'.format(conf["dataset"])
    model_state_file = model_path + model_name
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    log_path = './logs/{}/'.format(conf["dataset"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.INFO, filename=log_path+model_name+'.log')

    run_path = "./runs/{}/{}".format(conf["dataset"], model_name)
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    logging.info("Sanity Check: stat name : {}".format(model_state_file))
    run_path = "./runs/{}/{}".format(conf["dataset"], model_name)
    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    run = SummaryWriter(run_path)

    # build model
    model = REGCN(conf)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["wd"])

    # start training
    print("-----------------------------start training-------------------------------n")
    best_val_mrr, best_test_mrr = 0, 0
    accumulated = 0
    epoch_times = []
    for epoch in range(conf["n_epochs"]):
        epoch_start_time = time.time()   
        model.train()

        idx = [_ for _ in range(len(train_list))]
        random.shuffle(idx)
        
        losses = []
        hist_len = conf["hist_len"]
        epoch_anchor = epoch * len(idx)
        for batch_idx, train_sample_num in enumerate(tqdm(idx)):
            if train_sample_num == 0: 
                continue

            if train_sample_num - hist_len < 0:
                hist_list = train_times[0:train_sample_num]
            else:
                hist_list = train_times[train_sample_num - hist_len: train_sample_num]

            g_list = [graph_dict[tim].to(device) for tim in hist_list]
            output = torch.LongTensor(train_list[train_sample_num]).to(device)
            loss = model(g_list, output)
    

            losses.append(loss.item())

            batch_anchor = epoch_anchor + batch_idx
            run.add_scalar('loss/loss', loss.item(), batch_anchor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf["grad_norm"])  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        total_epoch_time = sum(epoch_times)

        logging.info(f'Epoch {epoch + 1}/{conf["n_epochs"]}, Time: {epoch_time:.2f}s, AvgTime: {average_epoch_time:.2f}s, TotTime: {total_epoch_time:.2f}s, Loss: {np.mean(losses)}')

        # validation and test
        if (epoch + 1) % conf["test_interval"] == 0:
            val_res = test_sparsity(conf,
                                model, 
                                model_state_file,
                                history_times,
                                val_times,
                                graph_dict,
                                valid_list,
                                all_ans_list_valid)
            
            mrr_filter_val,hits_filter_val = val_res["filter"]
            
            run.add_scalar('val/filter/overall mrr-ent', mrr_filter_val, epoch)

            test_res = test_sparsity(conf,
                                    model, 
                                    model_state_file,
                                    history_times,
                                    test_times,
                                    graph_dict,
                                    test_list, 
                                    all_ans_list_test)

            mrr_filter, hits_filter = test_res["filter"]

            mrr_val = mrr_filter_val
            mrr_test = mrr_filter
            if mrr_val < best_val_mrr:
                accumulated += 1
                if epoch >= conf["n_epochs"]:
                    print("Max epoch reached! Training done.")
                    break
                if accumulated >= conf["patience"]:
                    print("Early stop triggered! Training done at epoch{}".format(epoch))
                    break
            else:
                accumulated = 0
                best_val_mrr = mrr_val
                best_test_mrr = mrr_test
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        print(f"Epoch {epoch:04d}, Valid: MRR: {mrr_filter_val}, HITS: {hits_filter_val}")
        print(f"Epoch {epoch:04d}, Test: MRR: {mrr_filter}, HITS: {hits_filter}")
        print("Epoch {:04d}, AveLoss: {:.4f}, BestMRR: {:.4f}, Model: {}, Dataset: {}".format(epoch, np.mean(losses), best_test_mrr, conf["model"], conf["dataset"]))

    test_res = test_sparsity(conf,
                            model, 
                            model_state_file,
                            history_times,
                            test_times,
                            graph_dict,
                            test_list, 
                            all_ans_list_test, 
                            mode="test")
    print(f"Final Result: MRR: {test_res['filter'][0]}, HITS: {test_res['filter'][1]}")

                


if __name__ == '__main__':
    main()
