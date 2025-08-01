import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres
import sklearn
from sklearn import metrics
import sklearn.metrics as sk_metrics
from torch.nn import Parameter
from sklearn.neighbors import NearestNeighbors


class Clustering_Metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0,0,0,0,0,0,0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, ari, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)        
        outputs = self.activation(x)
        return outputs

def purity_score(y_true, y_pred):
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def generate_unconflicted_data_index(p, beta_1, beta_2):
    unconf_indices = []
    conf_indices = []
    p = p.detach().cpu().numpy()
    confidence1 = p.max(1)
    confidence2 = np.zeros((p.shape[0],))
    a = np.argsort(p, axis=1)[:,-2]
    for i in range(p.shape[0]):
        confidence2[i] = p[i,a[i]]
        if (confidence1[i] > beta_1) and (confidence1[i] - confidence2[i]) > beta_2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def generate_sep_index(emb, centers, p):
    emb = emb.detach().cpu().numpy()
    centers = centers.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(emb)
    _, indices = nn.kneighbors(centers)
    indices_sep = np.zeros((emb.shape[0], centers.shape[0]-2), dtype=int)
    assignments_index = np.argsort(p, axis=1)
    first_center_index = indices[assignments_index[:,-1]]
    second_center_index = indices[assignments_index[:,-2]]
    for i in range(emb.shape[0]):
        k = 0
        for j in indices:
            if (j != first_center_index[i]) and (j != second_center_index[i]):
                indices_sep[i, k] = j
                k+=1
    return indices_sep

def negative_embeddings(z_mu_pos, z_sigma2_log_pos, emb_pos):
    idx = torch.randperm(emb_pos.shape[0])
    z_mu_neg = z_mu_pos[idx,:]
    z_sigma2_log_neg = z_sigma2_log_pos[idx,:]
    emb_neg = emb_pos[idx,:]
    return z_mu_neg, z_sigma2_log_neg, emb_neg

def target_distribution(p, unconflicted_ind, conflicted_ind):
    p = p.detach().cpu().numpy()
    q = np.zeros(p.shape)
    q[conflicted_ind] = p[conflicted_ind]
    q[unconflicted_ind, np.argmax(p[unconflicted_ind], axis=1)] = 1
    q = torch.tensor(q, dtype=torch.float32).to("cuda:0")
    return q

def evaluate_links(adj, labels):
    count_links = {"nb_links": 0,
                   "nb_false_links": 0,
                   "nb_true_links": 0}
    for i, line in enumerate(adj):
        for j in range(line.indices.size):
            if line.indices[j] > i:
                count_links["nb_links"] += 1
                if labels[i] == labels[line.indices[j]]:
                    count_links["nb_true_links"] += 1
                else:
                    count_links["nb_false_links"] += 1
    return count_links
