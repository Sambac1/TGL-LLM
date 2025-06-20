import os
import numpy as np
import scipy.sparse as sp
import json
from torch.utils.data import Dataset, DataLoader
# import pickle

def _read_triplets(filename):
    with open(filename, 'r') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        id = int(triplet[4])
        st = int(triplet[3])
        # event_id = triplet[6]
        l.append([s, r, o, st, id])
    return l

def _read_candis_as_list(filename):
    l = []
    for triplet in _read_triplets(filename):
        id = int(triplet[0])
        candidates = eval(triplet[1])
        candidates = [int(x) for x in candidates]
        l.append([id]+candidates)
    return l


class Datasets():
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['data_path']
        self.name = conf['dataset']
        self.dir = os.path.join(self.path, self.name)
        self.device = conf["device"]
        batch_size_test = conf['batch_size_test']
        self.hist_len = conf['hist_len']
        self.num_sample_train = conf['num_sample_train']
        self.k = conf['num_candidate']
        self.rs = conf['train_sample']
        self.dataset = conf['dataset']

        self.num_ent,self.num_rel = self.get_ent_rel_size()
        train_data = np.array(_read_triplets_as_list(os.path.join(self.dir, 'train.txt'), load_time=True))
        valid_data = np.array(_read_triplets_as_list(os.path.join(self.dir, 'valid.txt'), load_time=True))
        test_data = np.array(_read_triplets_as_list(os.path.join(self.dir, 'test.txt'), load_time=True))
        train_candidates = np.array(_read_candis_as_list(os.path.join(self.dir, 'candidates','K_'+str(self.k),'train_'+str(self.k)+'_candidates.txt')))
        valid_candidates = np.array(_read_candis_as_list(os.path.join(self.dir, 'candidates','K_'+str(self.k),'valid_'+str(self.k)+'_candidates.txt')))
        test_candidates = np.array(_read_candis_as_list(os.path.join(self.dir, 'candidates','K_'+str(self.k),'test_'+str(self.k)+'_candidates.txt')))
        #+'_test'
        print("# Sanity Check:  train edges: {}".format(len(train_data)))

        train_split_by_time = self.split_by_time(train_data, 'train')
        valid_split_by_time = self.split_by_time(valid_data, 'valid')
        test_split_by_time = self.split_by_time(test_data, 'test')

        train_times = np.array(sorted(set(train_data[:, 3])))
        valid_times = np.array(sorted(set(valid_data[:, 3])))
        test_times = np.array(sorted(set(test_data[:, 3])))
        history_times = np.concatenate((train_times, valid_times, test_times), axis=None)

        if self.rs:
            IF_sample = "./data/{}/sample_10w_coreset_{}.json".format(conf['dataset'],conf['dataset'])
 
            with open(IF_sample,'r') as f:
                select_id = json.load(f)
            train_sample_s = train_data[np.isin(train_data[:, 4], select_id)]
            train_sample = train_sample_s[train_sample_s[:,3]>=train_times[self.hist_len]]
            print(len(train_sample))
            train_candidates_sample = train_candidates[np.isin(train_candidates[:, 0], select_id)]
            train_candidates_sample = train_candidates_sample[train_sample_s[:,3]>=train_times[self.hist_len]]
            print(len(train_candidates_sample))
        else:
            start_id = len(train_data[train_data[:,3]<train_times[self.hist_len]])
            train_sample,train_candidates_sample = self.select_train_sample(train_data,train_candidates,start_id)

        test_sample = test_data
        valid_sample = valid_data
        test_candidates_sample = test_candidates
        valid_candidates_sample = valid_candidates

        self.train_dataset = RGCNLinkTrainDataset(train_sample, train_candidates_sample, train_split_by_time, train_times, self.hist_len)
        self.test_dataset = RGCNLinkTestDataset(test_sample, test_candidates_sample, test_split_by_time,history_times, self.hist_len)
        self.valid_dataset = RGCNLinkTestDataset(valid_sample, valid_candidates_sample, valid_split_by_time,history_times, self.hist_len)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size_test, shuffle=True, num_workers=10)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size_test, shuffle=True, num_workers=20)

    def get_ent_rel_size(self):
        with open(os.path.join(self.dir, 'state.txt'), 'r') as f:
            line = f.readline()
            num_nodes, num_rels = line.strip().split("\t")
            num_nodes = int(num_nodes)
            num_rels = int(num_rels)
        print("# Sanity Check:  entities: {}".format(num_nodes))
        print("# Sanity Check:  relations: {}".format(num_rels))
        return num_nodes,num_rels
    
    def split_by_time(self, data, type):
        time_dict = dict()

        for row in data:
            time = row[3]  # Get the time value
            if time not in time_dict:
                time_dict[time] = []
            time_dict[time].append(np.delete(row, 3))

        for time in time_dict:
            time_dict[time] = np.array(time_dict[time])

        snapshot_list = list(time_dict.values())
        nodes = []
        rels = []
        for snapshot in snapshot_list:
            uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
            uniq_r = np.unique(snapshot[:,1])
            edges = np.reshape(edges, (2, -1))
            nodes.append(len(uniq_v))
            rels.append(len(uniq_r)*2)
        print("# Sanity Check:  {} ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}"
            .format(type, np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list])))
        
        return snapshot_list

    def select_train_sample(self,data,data_candidates, start_id):
        data = data[start_id:]
        data_candidates = data_candidates[start_id:]
        data_candidates_sample =data_candidates
        data_sample = data

        num_sample = self.num_sample_train
        row_rand = np.arange(data_sample.shape[0])
        np.random.seed(13)
        np.random.shuffle(row_rand)
        select_sample_id = row_rand[:num_sample]
        data_sample = data_sample[select_sample_id]
        data_candidates_sample = data_candidates_sample[select_sample_id]
        
        return data_sample,data_candidates_sample
    

class RGCNLinkTrainDataset(Dataset):
    def __init__(self, data, data_candidates, data_split_by_time, times, hist_len):
        self.data = data
        self.data_candidates = data_candidates
        self.data_split_by_time = data_split_by_time
        self.times = times
        self.times_list = list(times)
        self.hist_len = hist_len

    def __getitem__(self, idx):
        # index
        time = self.data[idx][3]
        time_idx = self.times_list.index(time)
        if time_idx - self.hist_len < 0:
            hist_list = self.times[0:time_idx]
        else:
            hist_list = self.times[time_idx - self.hist_len: time_idx]

        return self.data[idx], hist_list, self.data_candidates[idx]
    
    def __len__(self):
        return len(self.data)


class RGCNLinkTestDataset(Dataset):
    def __init__(self, data, data_candidates, data_split_by_time, times, hist_len):
        self.data = data
        self.data_candidates = data_candidates
        self.data_split_by_time = data_split_by_time
        self.times = times
        self.times_list = list(times)
        self.hist_len = hist_len

    def __getitem__(self, idx):
        time = self.data[idx][3]
        time_idx = self.times_list.index(time)
        if time_idx - self.hist_len < 0:
            hist_list = self.times[0:time_idx]
        else:
            hist_list = self.times[time_idx - self.hist_len: time_idx]

        
        return self.data[idx],hist_list,self.data_candidates[idx]
    
    def __len__(self):
        return len(self.data)



