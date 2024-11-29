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
np.seterr(divide='ignore', invalid='ignore')


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# 归一化特征
# 按行求均值
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


# 归一化邻接矩阵
# AD^{-1/2}.TD^{-1/2}
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def load_data2(path="../dataset/citeseer", dataset_str="citeseer"):
    print('Loading {} dataset...'.format(dataset_str))
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # 获取整个图的所有节点特征
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)  # 根据自己需要归一化特征
    features = torch.FloatTensor(np.array(features.todense()))

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = preprocess_adj(adj)  # 根据自己需要归一化邻接矩
    adj = torch.FloatTensor(np.array(adj.todense()))

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)
    #
    # label_dict = {}
    # for i in range(len(labels)):
    #     label_dict[i] = labels[i]
    # with open('citeseer_label_dict.csv', 'w') as f:
    #     [f.write('{0}  {1}\n'.format(key, value)) for key, value in label_dict.items()]
    #
    labels = torch.LongTensor(labels)

    # 划分训练集、验证集、测试集索引
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))
    idx_test = test_idx_range.tolist()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    x = I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
    return x


def get_adjacency_edge2(distance_df_filename='../dataset/citeseer/citeseer.csv'):
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    # print(len(df))  # 边个数为4731
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = x.tolist()
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
    print("打印citeseer_Ae")
    A_e = torch.from_numpy(A_e)
    print("打印citeseer_A_location_lp")
    A_location = np.float32(A_e > 0)
    A_location_lp = normalized_laplacian(A_location)
    A_location_lp = np.nan_to_num(A_location_lp)
    A_location_lp = A_location_lp.astype(np.float32)
    A_location_lp = torch.from_numpy(A_location_lp)
    return A_e, A_location_lp


def get_incidence_matrix2(distance_df_filename='../dataset/citeseer/citeseer.csv', num_of_vertices=3327, path="../dataset/citeseer/", dataset="citeseer"):
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    # print(len(df))  # 边个数为4731
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = x.tolist()
        list_index.append(x)
    node_matrix = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    node_number = node_matrix[:,0]
    # print(node_number)
    node_dict = {}
    n = 0
    for i in range(len(node_number)): #长度为3312
        node_dict[node_number[i]] = n
        n = n+1
    node_dict["197556"] = 3312
    node_dict["ghani01hypertext"] = 3313
    node_dict["38137"] = 3314
    node_dict["95786"] = 3315
    node_dict["nielsen00designing"] = 3316
    node_dict["flach99database"] = 3317
    node_dict["khardon99relational"] = 3318
    node_dict["kohrs99using"] = 3319
    node_dict["raisamo99evaluating"] = 3320
    node_dict["wang01process"] = 3321
    node_dict["hahn98ontology"] = 3322
    node_dict["tobies99pspace"] = 3323
    node_dict["293457"] = 3324
    node_dict["gabbard97taxonomy"] = 3325
    node_dict["weng95shoslifn"] = 3326
    # print(node_dict)  # 给每个节点编号重新弄了索引

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

