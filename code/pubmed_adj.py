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


def get_adjacency_edge3(distance_df_filename='../data/pubmed/pubmed.csv'):
    print("开始进行第二步...")
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
    np.savetxt('pubmed_edge_adj.txt', A_e)


if __name__ == '__main__':
    get_adjacency_edge3()