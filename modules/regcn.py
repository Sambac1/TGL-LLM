
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import ConvTransE

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, comp='sub'):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.skip_connect = skip_connect
        self.comp = comp

        # WL
        self.weight_neighbor = self.get_param([in_feat, out_feat])

        if self.self_loop:
            self.loop_weight = self.get_param([in_feat, out_feat])

        if self.skip_connect:
            self.skip_connect_weight =self.get_param([in_feat, out_feat])
            self.skip_connect_bias = self.get_param([out_feat])
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        return param


    def forward(self, g, prev_h, emb_rel):
        # should excecute before the propagate
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        node_repr = self.propagate(g, emb_rel)

        if self.self_loop:
            node_repr = node_repr + loop_message
        if prev_h is not None and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr

        return node_repr


    def propagate(self, g, emb_rel):
        g.update_all(lambda x: self.msg_func(x, emb_rel), fn.sum(msg='msg', out='h'), self.apply_func)
        return g.ndata['h']


    def msg_func(self, edges, emb_rel):
        relation = emb_rel.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        if self.comp == "sub":
            msg = node + relation
        elif self.comp == "mult":
            msg = node * relation
        msg = torch.mm(msg, self.weight_neighbor)

        return {'msg': msg}


    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, n_bases=-1, n_layers=1, dropout=0, act=F.rrelu, self_loop=False, skip_connect=False, comp="sub"):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.n_bases = n_bases
        self.n_layers = n_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.act = act
        self.comp = comp
        self.build_model()


    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            if self.skip_connect:
                sc = False if idx == 0 else True
            else:
                sc = False
            h2h = RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.n_bases, activation=self.act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, comp=self.comp)
            self.layers.append(h2h)


    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        prev_h = None
        for i, layer in enumerate(self.layers):
            prev_h = layer(g, prev_h, r)

        return prev_h
    

class REGCN(nn.Module):
    def __init__(self, conf):
        super(REGCN, self).__init__()

        self.conf = conf
        num_ents = conf["num_ent"]
        num_rels = conf["num_rel"]
        h_dim = conf["h_dim"]


        self.ent_embs = torch.nn.Parameter(torch.FloatTensor(num_ents, h_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.ent_embs)

        self.rel_embs = torch.nn.Parameter(torch.FloatTensor(num_rels, h_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.rel_embs)


        self.rgcn = RGCN(num_ents,
                             h_dim,
                             h_dim,
                             num_rels,
                             n_bases=conf["n_bases"],
                             n_layers=conf["n_layers"],
                             dropout=conf["dropout"],
                             self_loop=conf["self_loop"],
                             skip_connect=conf["skip_connect"])

        self.hist_gru = nn.GRU(h_dim, h_dim, batch_first=True) 

        # decoder
        self.decoder = ConvTransE(num_ents, h_dim, conf["input_dropout"], conf["hidden_dropout"], conf["feat_dropout"])

        self.loss_e = torch.nn.CrossEntropyLoss()


    def predict(self, g_list, triplets):
        all_triplets = triplets

        ent_embs = self.ent_embs
        rel_embs = self.rel_embs

        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)

        scores, query_emb, obj_emb = self.decoder.forward_cl(ent_rep, self.rel_embs, all_triplets)

        # return scores, all_triplets, query_emb, obj_emb,ent_rep
        return scores, history_embs, query_emb, rel_embs, ent_embs


    def predict_p(self, g_list, triplets):
        all_triplets = triplets

        ent_embs = self.ent_embs
        rel_embs = self.rel_embs

        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)

        scores, query_emb, obj_emb = self.decoder.forward_cl(ent_rep, self.rel_embs, all_triplets)

        return scores, all_triplets, query_emb, obj_emb,ent_rep
    

    def forward(self, g_list, triplets):
        scores, all_triplets, _, _,_ = self.predict_p(g_list, triplets)
        scores = scores.view(-1, self.conf["num_ent"])
        loss = self.loss_e(scores, all_triplets[:, 2])

        return loss
