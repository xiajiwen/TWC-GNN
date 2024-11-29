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


def cootootensor(edge_index_temp):
    # print(edge_index_temp)
    # 需要转换成tensor才能放在模型中运行
    values = edge_index_temp.data  # 边上对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式
    # print(edge_index_A)
    i = torch.LongTensor(indices)  # 转tensor
    v = torch.FloatTensor(values)  # 转tensor
    edge_index = torch.sparse.FloatTensor(i, v, edge_index_temp.shape)
    return edge_index


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


def load_data3(path="../dataset/pubmed", dataset_str="pubmed"):
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

    #
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
    # adj = torch.FloatTensor(np.array(adj.todense()))
    adj = cootootensor(adj.tocoo())

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # label_dict = {}
    # for i in range(len(labels)):
    #     label_dict[i] = labels[i]
    # with open('pubmed_label_dict.csv', 'w') as f:
    #     [f.write('{0}  {1}\n'.format(key, value)) for key, value in label_dict.items()]

    labels = torch.LongTensor(labels)

    # 划分训练集、验证集、测试集索引
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))
    idx_test = test_idx_range.tolist()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    dataset = Planetoid(root='Pubmed', name='Pubmed')
    # x = dataset[0].x
    edge_index = dataset[0].edge_index
    print("数据加载完成...")
    return edge_index, features, labels, idx_train, idx_val, idx_test


def normalized_laplacian(mx):
    """Row-normalize sparse matrix"""
    # qiu du juzhen
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # I-D(-1/2)*A*D(-1/2)
    I = sp.eye(mx.shape[0])
    a1 = r_mat_inv.dot(mx)
    a2 = a1.dot(r_mat_inv)
    a3 = I - a2
    return a3


def get_adjacency_edge3(distance_df_filename='../dataset/pubmed/pubmed.csv'):
    print("开始进行第二步...")
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')     # print(len(df))  # 边个数为4731
    list_index = []     # 第一步：边（node1,node2）作为索引放入list中
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = x.tolist()
        list_index.append(x)

    list_out = df['from'].values.tolist()    # 第二步：生成出度和入读字典
    list_in = df['to'].values.tolist()
    dict_out = {x: list_out.count(x) for x in set(list_out)}
    dict_in = {x: list_in.count(x) for x in set(list_in)}

    dict_all = copy.deepcopy(dict_out)      # 第三步 生成度信息（出度+入度）求ct
    for k, v in dict_in.items():
        if k in dict_all.keys():
            dict_all[k] = dict_all[k] + v
        else:
            dict_all[k] = v
    dict_all = sorted(dict_all.items(), key=lambda item:item[0])  # 排序方便看，生成的是元组
    dict_all = dict(dict_all)  # 变回字典
    list_du = []     # 3.1求方差 求ct
    for v in dict_all.values():
        list_du.append(v)
    arr_du = np.array(list_du)
    var = np.var(arr_du)
    A_e = np.zeros((int(len(df)), int(len(df))),    # 根据出度入度字典生成边邻接矩阵
                 dtype=np.float32)
    for i in range(len(df)):
        for j in range(len(df)):           # a,b节点都到达c节点 竞争关系
            if list_index[i][1] == list_index[j][1]:        # 看a,b出度的影响
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
    # print("加载txt文件")
    # A_e = np.loadtxt('./pubmed_edge_adj.txt')
    print("已经生成好pubmed边的（有权重）邻接矩阵，现在是numpy.ndarray")
    # 下一步将ndarray转成coo稀疏矩阵
    edge_index_temp = sp.coo_matrix(A_e)
    # 需要转换成tensor才能放在模型中运行
    edge_index = cootootensor(edge_index_temp)
    print("有权重的tensor邻接矩阵已经完成")
    print("开始操作无权重的邻接矩阵，并且需要拉普拉斯矩阵")
    A_location = np.float32(A_e > 0)
    lapu_edge_coo_np = sp.coo_matrix(A_location)
    lapu_edge_coo_lp_np = normalized_laplacian(lapu_edge_coo_np)
    # 需要转换成tensor才能放在模型中运行
    lapu_edge_coo_lp_np = lapu_edge_coo_lp_np.tocoo()
    lapu_edge_coo_lp_tens = cootootensor(lapu_edge_coo_lp_np)
    print("拉普拉斯矩阵已经完成")
    return edge_index, lapu_edge_coo_lp_tens


def get_incidence_matrix3(distance_df_filename='../dataset/pubmed/pubmed.csv', num_of_vertices=19717):
    print("开始进行第三步...")
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    # print(len(df))  # 边个数为44338
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = x.tolist()
        list_index.append(x)
    print(list_index)
    # 将19717个paper名字放入list中
    with open('../dataset/pubmed/pubmed_node.txt', 'r') as f:
        list_node = f.read().splitlines()
    list_node = [int(x) for x in list_node]
    # 给19717个paper重新编号
    node_dict = {}
    n = 0
    for i in range(len(list_node)):
        node_dict[list_node[i]] = n
        n = n+1
    print(node_dict)  # 给每个节点编号重新弄了索引

    A = np.zeros((int(num_of_vertices), int(len(df))),
                   dtype=np.float32)
    for i in range(len(df)):
        x1 = node_dict[list_index[i][0]]
        x2 = node_dict[list_index[i][1]]
        A[x1][i] = 1
        A[x2][i] = -1
    print("开始将关联矩阵变为稀疏矩阵")
    edge_index_temp = sp.coo_matrix(A)
    # 需要转换成tensor才能放在模型中运行
    A_guanlian = cootootensor(edge_index_temp)
    print("关联矩阵已经变为稀疏矩阵")
    return A_guanlian


# def load_data(path="./cora/Cora/raw", dataset_str="cora"):
#     """Load data."""
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     # normalize
#     features = normalize(features)
#
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     # network_emb = pros(adj)
#     # network_emb = 0
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot
#
#     idx_train = range(len(y))  # training data index
#     idx_val = range(len(y), len(y) + 500)  # validation data index
#     idx_test = test_idx_range.tolist()  # test data index
#
#     features = np.array(features.todense())
#     features = torch.FloatTensor(features)
#     adj = torch.FloatTensor(np.array(adj.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#     return adj, features, labels, idx_train, idx_val, idx_test
#
#
# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index
#
#
# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     retu
