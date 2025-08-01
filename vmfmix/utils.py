# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment

from scipy.special import iv
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, \
        adjusted_mutual_info_score as AMI, adjusted_rand_score as AR, silhouette_score as SI, \
        calinski_harabasz_score as CH


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w


def predict(x, mu, k, pi, n_cluster):

    yita_c = np.exp(np.log(pi[np.newaxis, :]) + blmm_pdfs_log(x, k, mu, n_cluster))

    yita = yita_c
    return np.argmax(yita, axis=1)

def predict_pro(x, mu, k, pi, n_cluster):

    yita_c = np.exp(np.log(pi[np.newaxis, :]) + vmf_pdfs_log(x, k, mu, n_cluster))

    yita = yita_c / np.sum(yita_c, axis=1, keepdims=True)
    return yita

def vmf_pdfs_log(x, ks, mus, n_cluster):

    VMF = []
    for c in range(n_cluster):
        VMF.append(vmf_pdf_log(x, mus[c:c+1], ks[c]).reshape(-1, 1))
    return np.concatenate(VMF, 1)


def vmf_pdf_log(x, mu, k):

    D = x.shape[len(x.shape) - 1]
    pdf = (D / 2 - 1) * np.log(k) - (D / 2) * np.log(2 * np.pi) - np.log(iv(D / 2 - 1, k)) + x.dot(mu.T * k)
    return pdf



def blmm_pdfs_log(x, ks, mus, n_cluster):

    VMF = []
    for c in range(n_cluster):
        VMF.append(blmm_pdf_log(x, mus[c:c+1, :], ks[c]).reshape(-1, 1))
    return np.concatenate(VMF, 1)


def blmm_pdf_log(x, mu, k):

    D = x.shape[len(x.shape) - 1]
    pdf = (D / 2 - 1) * np.log(k) - (D / 2) * np.log(2 * np.pi) - np.log(iv(D / 2 - 1, k)) + x.dot(mu.T * k)
    return pdf


def calculate_mix(a, b, K):

    lambda_bar = a / (a + b)
    pi = np.zeros(K,)
    for i in range(K):
        temp_temp = 1
        for j in range(i):
            temp = 1 - (lambda_bar[j])
            temp_temp = temp_temp * temp
        pi[i] = lambda_bar[i] * temp_temp
    return pi


def d_besseli(nu, kk):

    try:
        bes = iv(nu + 1, kk) / (iv(nu, kk) + 1e-10) + nu / (kk+1e-10)
        assert (min(np.isfinite(bes)))
    except:
        bes = np.sqrt(1 + (nu**2) / (kk**2))

    return bes

# def d_besseli(v, z, eps=1e-20):
#     # Ensure eps is a compatible type with z
#     eps_array = np.array(eps, dtype=z.dtype)
#
#     def delta_a(a):
#         lamb = v + (a - 1.0) / 2.0
#         return (v - 0.5) + lamb / (
#             2 * np.sqrt(np.maximum(np.power(lamb, 2) + np.power(z, 2), eps_array))
#         )
#
#     delta_0 = delta_a(np.array(0.0, dtype=z.dtype))
#     delta_2 = delta_a(np.array(2.0, dtype=z.dtype))
#     B_0 = z / (
#         delta_0 + np.sqrt(np.maximum(np.power(delta_0, 2) + np.power(z, 2), eps_array))
#     )
#     B_2 = z / (
#         delta_2 + np.sqrt(np.maximum(np.power(delta_2, 2) + np.power(z, 2), eps_array))
#     )
#
#     return (B_0 + B_2) / 2.0


# def d_besseli_low(nu, kk):
#
#     try:
#         bes = iv(nu + 1, kk) / (iv(nu, kk) + np.exp(-700)) + nu / kk
#         assert (min(np.isfinite(bes)))
#     except:
#         bes = kk / (nu + 1 + np.sqrt(kk**2 + (nu + 1)**2)) + nu / kk
#
#     return bes

def d_besseli_low(v, z, eps=1e-20):
    # Ensure eps is a compatible type with z
    eps_array = np.array(eps, dtype=z.dtype)

    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * np.sqrt(np.maximum(np.power(lamb, 2) + np.power(z, 2), eps_array))
        )

    delta_0 = delta_a(np.array(0.0, dtype=z.dtype))
    delta_2 = delta_a(np.array(2.0, dtype=z.dtype))
    B_0 = z / (
        delta_0 + np.sqrt(np.maximum(np.power(delta_0, 2) + np.power(z, 2), eps_array))
    )
    B_2 = z / (
        delta_2 + np.sqrt(np.maximum(np.power(delta_2, 2) + np.power(z, 2), eps_array))
    )

    return (B_0 + B_2) / 2.0


def caculate_pi(model, N, T):

    resp = np.zeros((N, T))
    resp[np.arange(N), model.labels_] = 1
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    pi = (nk / N)[np.newaxis, :]
    return pi


def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return v, log_norm

def console_log(pred, data=None, labels=None, model_name='cluster', newJ=None, verbose=1):

    measure_dict = dict()
    if data is not None:
        # measure_dict['si'] = SI(data, pred)
        measure_dict['ch'] = CH(data, pred)
    if labels is not None:
        measure_dict['acc'] = cluster_acc(pred, labels)[0]
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['ar'] = AR(labels, pred)
        measure_dict['ami'] = AMI(labels, pred)
    if newJ is not None:
        measure_dict['new_component'] = newJ

    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict