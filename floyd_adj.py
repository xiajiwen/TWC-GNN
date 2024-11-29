import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import math


def get_sp_adj(distance_df_filename='./data/citeseer/citeseer_good.csv', num_of_vertices=3312, path="./data/citeseer/", dataset="citeseer"):
    with open(distance_df_filename, 'r') as f:
        df = pd.read_csv(f, engine='python')
    print(len(df))  # 边个数
    # 第一步：边（node1,node2）作为索引放入list中
    list_index = []
    for i in range(len(df)):
        x = df.loc[i].values[:]
        x = x.tolist()
        list_index.append(x)
    print(list_index)
    print(len(list_index))
    # 第二步给每个节点编号
    node_matrix = np.genfromtxt("{}{}.content".format(path, dataset),
                                dtype=np.dtype(str))
    node_number = node_matrix[:, 0]
    node_dict = {}
    n = 0
    for i in range(len(node_number)):  # 长度为2708
        node_dict[node_number[i]] = n
        n = n + 1
    print(node_dict)  # 给每个节点编号重新弄了索引
    # 第三步建立邻接矩阵
    A_e = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    for i in range(len(df)):
        print(list_index[i][0], node_dict[list_index[i][0]])
        print(list_index[i][1],node_dict[list_index[i][1]])
        A_e[node_dict[list_index[i][0]]][node_dict[list_index[i][1]]] = 1
    print(A_e)
    ilter = [i for i in range(num_of_vertices)]
    for o in ilter:
        for d in ilter:
            if d == o:
                continue
            if A_e[o, d] == 0:
                A_e[o, d] = 999
    print(A_e)
    for mid in ilter:
        if mid % 10 == 0:
            print("进度~~%d%d" % (mid, num_of_vertices))
        for o in ilter:
            for d in ilter:
                if A_e[o, mid] != 999 and A_e[mid, d] != 999 and A_e[o, d] > A_e[o, mid] + A_e[mid, d]:
                    A_e[o, d] = A_e[o, mid] + A_e[mid, d]
    print(A_e)
    np.savetxt('floyd_adj_citeseer.txt', A_e)
    return A_e


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_sp_adj2(num_of_vertices=3312, path="./data/citeseer/", dataset="citeseer"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    edges_unordered_A = np.genfromtxt("{}citeseer_A.txt".format(path, dataset), dtype=np.int32)
    edges_A = np.array(edges_unordered_A[:, 0:2])
    edges_A = edges_A - 1
    A_e = sp.coo_matrix((edges_unordered_A[:, 2], (edges_A[:, 0], edges_A[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    A_e = A_e.todense()
    # to_dense()
    print("kaishi")
    ilter = [i for i in range(num_of_vertices)]
    for o in ilter:
        for d in ilter:
            if d == o:
                continue
            if A_e[o, d] == 0:
                A_e[o, d] = 999
    print(A_e)
    for mid in ilter:
        if mid % 10 == 0:
            print("进度~~%d%d" % (mid, num_of_vertices))
        for o in ilter:
            for d in ilter:
                if A_e[o, mid] != 999 and A_e[mid, d] != 999 and A_e[o, d] > A_e[o, mid] + A_e[mid, d]:
                    A_e[o, d] = A_e[o, mid] + A_e[mid, d]
    print(A_e)
    np.savetxt('floyd_adj_citeseer.txt', A_e)
    return A_e
if __name__ == '__main__':
    sp_adj = get_sp_adj()
