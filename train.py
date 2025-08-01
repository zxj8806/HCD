import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math
import itertools
from model import HCD
from preprocessing import load_data, mask_test_edges, sparse_to_tuple, preprocess_graph
from sklearn.metrics import confusion_matrix

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def map_vector_to_clusters(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    y_true_mapped = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        y_true_mapped[i] = col_ind[y_true[i]]
    return y_true_mapped.astype(int)

dataset = "Cora"
adj, features, labels = load_data('cora', './data/Cora')
nClusters = 7

alpha = 1.
gamma_1 = 1.
gamma_2 = 1.
gamma_3 = 1.
num_neurons = 32
embedding_size = 64
save_path = "./results/"

adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()


adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to("cuda:0")
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).to("cuda:0")
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2])).to("cuda:0")
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig
weight_tensor_orig = weight_tensor_orig.to("cuda:0")


network = HCD(num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU", alpha=alpha, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3).to("cuda:0")
y_pred, y = network.train(features, adj_norm, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=1000, lr=0.01, save_path=save_path, dataset=dataset, run_id="run1")

target_names = ["0", "1", "2", "3", "4", "5", "6"]
y_mapped = map_vector_to_clusters(y, y_pred)
