import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphConvolution_edge, GraphConvolution, EncoderLayer
from torch_geometric.nn import GATConv

class MODEL(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nedge, n_layers, attention_dropout_rate):
        super(MODEL, self).__init__()
        self.dropout = dropout
        self.egc0 = GraphConvolution_edge(nfeat, nedge)  # 边维度nedge
        self.egc1 = GraphConvolution(nedge, nhid)  # 边维度nhid
        self.nheads = nheads
        encoders = [EncoderLayer(nedge, nhid, dropout, attention_dropout_rate, nheads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(nedge)  # 边维度nhid+nedge

        self.attentions = [GraphAttentionLayer(nfeat + nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 节点维度nhid * nheads
        self.out_att1 = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.gat = GATConv(nfeat + nhid, nhid, nheads, dropout=dropout)
        self.gat2 = GATConv(nhid * nheads, nclass, 1, dropout=dropout)

    def forward(self, x, adj, adj_e, M_guanlian, adj_location):
        ifPubmed = True
        if ifPubmed == True:
            # 转换成边信息
            z0 = self.egc0(x, M_guanlian)  # 对z0进行gcn （n边,nedge)

            # # 边transformer
            # for enc_layer in self.layers:
            #     z1 = enc_layer(z0, adj_location)  # 对z1进行transformer （n边,nedge)
            # z1 = self.final_ln(z1)

            z1 = F.relu(self.egc1(z0, adj_e))  # 对z0进行gcn （n边,nhid)
            # 转换成节点信息
            x1 = torch.mm(M_guanlian, z1)  # 节点维度 nhid
            h0 = torch.cat((x, x1), 1)  # 节点nhid+nfea
            h0 = F.dropout(h0, self.dropout, training=self.training)
            h1 = self.gat(h0, adj)  # 节点维度nhid*nheads
            h1 = self.gat2(h1, adj)  # nclass
            # h1 = torch.cat([att(h0, adj) for att in self.attentions], dim=1)  # 节点维度nhid * nheads
            # h1 = F.dropout(h1, self.dropout, training=self.training)
            # h1 = self.out_att1(h1, adj)  # 节点维度nclass
        else:
            z0 = self.egc0(x, M_guanlian)  # 对z0进行gcn （n边,nedge)
            # 边transformer
            for enc_layer in self.layers:
                z1 = enc_layer(z0, adj_location)  # 对z1进行transformer （n边,nedge)
            z1 = self.final_ln(z1)
            z2 = F.relu(self.egc1(z1, adj_e))  # 对z0进行gcn （n边,nhid)
            # 转换成节点信息
            x1 = torch.mm(M_guanlian, z2)  # 节点维度 nhid
            h0 = torch.cat((x, x1), 1)  # 节点nhid+nfea
            h0 = F.dropout(h0, self.dropout, training=self.training)
            # gat
            h1 = torch.cat([att(h0, adj) for att in self.attentions], dim=1)  # 节点维度nhid * nheads
            h1 = F.dropout(h1, self.dropout, training=self.training)
            h1 = self.out_att1(h1, adj)  # 节点维度nclass
        return F.log_softmax(h1, dim=1)


