from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataset_process_cora import accuracy
from dataset_process_cora import load_data, get_adjacency_edge, get_incidence_matrix
from dataset_process_citeseer import load_data2, get_adjacency_edge2, get_incidence_matrix2
from dataset_process_pubmed import load_data3, get_adjacency_edge3, get_incidence_matrix3
from models import MODEL

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--nedge', type=int, default=128, help='Number of hidden edges.')
parser.add_argument('--n_layers', type=int, default=3, help='n_layers.')
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)

paper = "citeseer"
if paper == "cora":
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    adj_e, adj_location = get_adjacency_edge()
    M_guanlian = get_incidence_matrix()
elif paper == "citeseer":
    adj, features, labels, idx_train, idx_val, idx_test = load_data2()
    adj_e, adj_location = get_adjacency_edge2()
    M_guanlian = get_incidence_matrix2()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data3()
    adj_e, adj_location = get_adjacency_edge3()
    M_guanlian = get_incidence_matrix3()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Model and optimizer
model = MODEL(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=int(labels.max()) + 1,
              dropout=args.dropout,
              nheads=args.nb_heads,
              alpha=args.alpha,
              nedge=args.nedge,
              n_layers=args.n_layers,
              attention_dropout_rate=args.attention_dropout_rate)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    adj_e = adj_e.cuda()
    M_guanlian = M_guanlian.cuda()
    adj_location = adj_location.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, adj_e, M_guanlian, adj_location)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, adj_e, M_guanlian, adj_location)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.load_state_dict(torch.load("./520.pkl"))
    model.eval()
    output = model(features, adj, adj_e, M_guanlian, adj_location)
    #
    # preds = output.max(1)[1].type_as(labels).cpu().numpy()
    # print(preds)
    # result_dict = {}
    # for i in range(len(preds)):
    #     result_dict[i] = preds[i]
    # with open('pubmed_result.csv', 'w') as f:
    #     [f.write('{0}  {1}\n'.format(key, value)) for key, value in result_dict.items()]
    #

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


# Testing
compute_test()
