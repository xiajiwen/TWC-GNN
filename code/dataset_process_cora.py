import numpy as np
import torch
import math
import copy
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys
from torch_geometric.datasets import Planetoid
np.seterr(divide='ignore', invalid='ignore')


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    print(features.shape)
    print(type(idx_features_labels[:, -1]))
    labels = encode_onehot(idx_features_labels[:, -1])  # (2708,7)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # idx_train = range(1624)
    # idx_val = range(1624, 2125)
    # idx_test = range(2125, 2625)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    x = I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
    return x


def get_adjacency_edge(distance_df_filename='../dataset/cora/cora.csv'):
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    # print(len(df))  # 边个数为5429
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = np.trunc(x).astype(int).tolist()
        list_index.append(x)
    # print(list_index)
    # print(len(list_index))
    # 第二步：生成出度和入读字典
    list_out = df['from'].values.tolist()
    list_in = df['to'].values.tolist()
    dict_out = {x: list_out.count(x) for x in set(list_out)}
    dict_in = {x: list_in.count(x) for x in set(list_in)}
    # print(dict_out)
    # print(dict_in)

    # 第三步 生成度信息（出度+入度）求ct
    dict_all = copy.deepcopy(dict_out)
    for k, v in dict_in.items():
        if k in dict_all.keys():
            dict_all[k] = dict_all[k] + v
        else:
            dict_all[k] = v
    dict_all = sorted(dict_all.items(), key=lambda item:item[0])  # 排序方便看，生成的是元组
    dict_all = dict(dict_all)  # 变回字典
    # print(dict_all)
    # 3.1求方差 求ct
    list_du = []
    for v in dict_all.values():
        list_du.append(v)
    # print(list_du)
    arr_du = np.array(list_du)
    # print(arr_du)
    var = np.var(arr_du)
    # print(var)

    # 根据出度入度字典生成边邻接矩阵
    A_e = np.zeros((int(len(df)), int(len(df))),
                 dtype=np.float32)
    for i in range(len(df)):
        for j in range(len(df)):
            if list_index[i][1] == list_index[j][1]:
                # a,b节点都到达c节点 竞争关系
                # 看a,b出度的影响
                a = dict_out[list_index[i][0]]
                b = dict_out[list_index[j][0]]
                ct = var
                x_jz = a+b-2
                x_jz2 = math.pow(x_jz, 2)/ct
                exp1 = np.exp(-x_jz2)
                A_e[i][j] = exp1

            elif list_index[i][0] == list_index[j][0]:  # a,b节点都从c节点出发 竞争关系
                a2 = dict_in[list_index[i][1]]
                b2 = dict_in[list_index[j][1]]
                ct2 = var
                x_jz2 = a2 + b2 - 2
                x_jz22 = math.pow(x_jz2, 2) / ct2
                exp2 = np.exp(-x_jz22)
                A_e[i][j] = exp2

            elif list_index[i][1] == list_index[j][0]:  # a-->c,c-->b 流通关系
                a3 = dict_out[list_index[i][1]]
                b3 = dict_in[list_index[i][1]]
                ct3 = var
                x_jz3 = a3 + b3 - 2
                x_jz23 = math.pow(x_jz3, 2) / ct3
                exp3 = np.exp(-x_jz23)
                A_e[i][j] = exp3

            elif list_index[i][0] == list_index[j][1]:  # c-->a,b-->c 流通关系
                a4 = dict_out[list_index[i][0]]
                b4 = dict_in[list_index[i][0]]
                ct4 = var
                x_jz4 = a4 + b4 - 2
                x_jz24 = math.pow(x_jz4, 2) / ct4
                exp4 = np.exp(-x_jz24)
                A_e[i][j] = exp4

            else:
                A_e[i][j] = 0

    for i in range(len(df)):
        A_e[i][i] = 0
    print("打印Ae")
    A_e = torch.from_numpy(A_e)

    print("打印A_location_lp")
    A_location = np.float32(A_e > 0)
    A_location_lp = normalized_laplacian(A_location)
    A_location_lp = np.nan_to_num(A_location_lp)
    A_location_lp = A_location_lp.astype(np.float32)
    A_location_lp = torch.from_numpy(A_location_lp)
    return A_e, A_location_lp


def get_incidence_matrix(distance_df_filename='../dataset/cora/cora.csv', num_of_vertices=2708, path="../dataset/cora/", dataset="cora"):
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    # print(len(df))  # 边个数为340
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = np.trunc(x).astype(int).tolist()
        list_index.append(x)
    node_matrix = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(int))
    node_number = node_matrix[:,0]
    node_dict = {}
    n = 0
    for i in range(len(node_number)): #长度为2708
        node_dict[node_number[i]] = n
        n = n+1
    # print(node_dict) # 给每个节点编号重新弄了索引
    A = np.zeros((int(num_of_vertices), int(len(df))),
                   dtype=np.float32)
    for i in range(len(df)):
        x1 = node_dict[list_index[i][0]]
        x2 = node_dict[list_index[i][1]]
        A[x1][i] = 1
        A[x2][i] = -1
    # print(A)
    A = torch.from_numpy(A)
    return A

