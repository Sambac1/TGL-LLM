
import torch
import numpy as np
import os

class RGCNLinkDataset(object):
    def __init__(self, name, dir=None):
        self.name = name
        self.dir = dir
        self.dir = os.path.join(self.dir, self.name)


    def load(self, load_time=True):
        stat_path = os.path.join(self.dir,  'state.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        self.train = np.array(_read_triplets_as_list(train_path, load_time))
        self.valid = np.array(_read_triplets_as_list(valid_path, load_time))
        self.test = np.array(_read_triplets_as_list(test_path, load_time))
        with open(os.path.join(self.dir, 'state.txt'), 'r') as f:
            line = f.readline()
            num_nodes, num_rels = line.strip().split("\t")
            num_nodes = int(num_nodes)
            num_rels = int(num_rels)
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        print("# Sanity Check:  edges: {}".format(len(self.train)))


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
        if load_time:
            st = int(triplet[3])
            # candidate = eval(triplet[6])
            # et = int(triplet[4])
            l.append([s, r, o, st])
            # l.append([s, r, o, st, et])
            # l.append([s, r, o, st] + candidate)
        else:
            l.append([s, r, o])
    return l

def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)

def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
        else:
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #

    return score


def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #

    return score


def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    #print(test_triples[1])
    #sparsity_tag = list(map(lambda x: sparsity_map(x, sparsity_split_ent, sparsity_split_rel), test_triples))
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    # print("mrr: {}".format(mrr))
    # print("rank: {}".format(rank))

    return filter_mrr.item(), mrr.item(), rank, filter_rank


def get_total_rank_sparse(test_triples, score, all_ans, eval_bz, predict_type=2):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, predict_type]

        rank.append(sort_and_rank(score_batch, target))

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())

    return filter_mrr.item(), mrr.item(), rank, filter_rank


def cal_ranks(rank_list, hit_ks=[1, 3, 10]):
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    hit_res = []
    for hit in hit_ks:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_res.append(avg_count)

    return mrr, hit_res
    # return mrr

def split_by_time(arr):
    time_dict = dict()

    for row in arr:
        time = row[3]  # Get the time value
        if time not in time_dict:
            time_dict[time] = []
        time_dict[time].append(np.delete(row, 3))

    # Convert lists of rows back into arrays
    for time in time_dict:
        time_dict[time] = np.array(time_dict[time])
    snapshot_list = list(time_dict.values())

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list