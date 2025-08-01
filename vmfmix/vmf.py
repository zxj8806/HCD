# -*- coding: utf-8 -*-
try:
    import numpy as np
    import scipy.io as scio
    import argparse
    import torch

    from scipy.special import digamma, iv, gammaln
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as NMI
    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.model_selection import train_test_split
    from numpy.matlib import repmat
    izip = zip

    from vmfmix.utils import cluster_acc, predict, calculate_mix, d_besseli, d_besseli_low, caculate_pi, log_normalize, console_log, predict_pro
except ImportError as e:
    print(e)
    raise ImportError

def l2_normalize(data):
    # Calculate the L2 norm of each data point
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    # Normalize the data
    normalized_data = data / norms
    return normalized_data


class VMFMixture:
    """
    Variational Inference Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, n_cluster, max_iter):

        self.T = n_cluster
        self.max_k = 0
        self.newJ = n_cluster
        self.max_iter = max_iter
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.pi = None

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None
        self.g = None
        self.h = None

        self.temp_zeta = None
        self.det = 1e-10
        #For MNIST/F-MNIST threshold = 0.01
        #For ImageNet 0.005 IMB 0.004
        self.threshold = 0.004

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.05,
            'u': 1,
            'v': 0.01,
            'gamma': 1,
        }

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        self.u = np.ones(self.T)
        self.v = np.ones(self.T) * 0.01
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.rho = np.ones((self.N, self.T)) * (1 / self.T)
        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)
        self.update_g_h(self.rho)

    def var_inf(self, x):

        D = self.D
        for ite in range(self.max_iter):
            # compute rho
            E_log_1_pi = np.roll(np.cumsum(digamma(self.h) - digamma(self.g + self.h)), 1)
            E_log_1_pi[0] = 0
            self.rho = x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T) + (D / 2 - 1) * \
                       (digamma(self.u) - np.log(self.v)) - \
                       (D / 2 * np.log(2 * np.pi)) - \
                       (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) - \
                       np.log(iv((D / 2 - 1), self.k) + np.exp(-700)) + \
                       digamma(self.g) - digamma(self.g + self.h) + E_log_1_pi
            if np.any(np.isnan(self.rho)):
                print(1)
            self.rho = np.exp(self.rho) / (np.expand_dims(np.sum(np.exp(self.rho), 1), 1) + 1e-8)
            if np.any(np.isnan(self.rho)):
                print(1)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)
            self.update_g_h(self.rho)

            if ite == self.max_iter-1:
                self.pi = calculate_mix(self.g, self.h, self.T)
                self.calculate_new_com()
                #print(1)

    def update_u_v(self, rho):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * np.sum(rho, 0) + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        if np.any(np.isnan(self.u)):
            print(1)
        self.v = self.prior['v'] + (d_besseli(D / 2 - 1, self.k)) * np.sum(rho, 0) \
                 + self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))
        if np.any(np.isnan(self.v)):
            print(1)

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + rho.T.dot(x)
        self.zeta = np.linalg.norm(temp, axis=1)
        if np.any(np.isnan(self.zeta)):
            print(1)
        self.xi = temp / self.zeta[:, np.newaxis]
        if np.any(np.isnan(self.xi)):
            print(1)

    def update_g_h(self, rho):
        # compute g, h
        N_k = np.sum(rho, 0)
        self.g = 1 + N_k
        for i in range(self.T):
            if i == self.T - 1:
                self.h[i] = self.prior['gamma']
            else:
                temp = rho[:, i + 1:self.T]
                self.h[i] = self.prior['gamma'] + np.sum(np.sum(temp, 1), 0)

    def calculate_new_com(self):

        threshold = self.threshold

        index = np.where(self.pi > threshold)[0]
        self.pi = self.pi[self.pi > threshold]
        self.newJ = self.pi.size

        self.xi = self.xi[index]
        self.k = self.k[index]
        self.u = self.u[index]
        self.v = self.v[index]
        self.zeta = self.zeta[index]

        # if self.newJ != self.T:
        #     print("new component of VMFMM is {}".format(self.newJ))

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)


    def warm_fit(self, data, initPi, initXi, initK):

        self.init_params(data)
        self.pi = initPi
        self.xi = initXi
        self.k = initK
        self.var_inf(data)


    def cold_start_fit(self, data):

        self.init_params(data)
        self.var_inf(data)
        return self

    def warm_start_fit(self, data):

        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        pred = predict(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        return pred

class CVMFMixture:
    """
    Collapsed Variational Inference Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, n_cluster, max_iter=100):

        self.T = n_cluster
        self.max_k = 0
        self.max_iter = max_iter
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.pi = None

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None

        self.temp_zeta = None
        self.det = 1e-10

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.01,
            'u': 0.5,
            'v': 0.01,
            'gamma': 1,
        }

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.rho = np.ones((self.N, self.T)) * (1 / self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)

    def caclulate_log_lik_x(self, x):

        D = self.D
        log_like_x = x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T) + (D / 2 - 1) * \
        (digamma(self.u) - np.log(self.v)) - \
        (D / 2 * np.log(2 * np.pi)) - \
        (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) - \
        np.log(iv((D / 2 - 1), self.k) + np.exp(-700))

        return log_like_x

    def caclulate_rho(self, x):

        gamma = self.prior['gamma']
        log_like_x = self.caclulate_log_lik_x(x)

        E_Nc_minus_n = np.sum(self.rho, 0, keepdims=True) - self.rho
        E_Nc_minus_n_cumsum_geq = np.fliplr(np.cumsum(np.fliplr(E_Nc_minus_n), axis=1))
        E_Nc_minus_n_cumsum = E_Nc_minus_n_cumsum_geq - E_Nc_minus_n

        first_tem = np.log(1 + E_Nc_minus_n) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        first_tem[:, self.T - 1] = 0
        dummy = np.log(gamma + E_Nc_minus_n_cumsum) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        second_term = np.cumsum(dummy, axis=1) - dummy

        rho = log_like_x + (first_tem + second_term)
        log_rho, log_n = log_normalize(rho)
        rho = np.exp(log_rho)

        return rho

    def var_inf(self, x):

        for ite in range(self.max_iter):
            # compute rho
            self.rho = self.caclulate_rho(x)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)

            if ite == self.max_iter - 1:
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k

    def update_u_v(self, rho):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * np.sum(rho, 0) + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        if np.any(np.isnan(self.u)):
            print(1)
        self.v = self.prior['v'] + (d_besseli(D / 2 - 1, self.k)) * np.sum(rho, 0) \
                 + self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))
        if np.any(np.isnan(self.v)):
            print(1)

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + rho.T.dot(x)
        self.zeta = np.linalg.norm(temp, axis=1)
        if np.any(np.isnan(self.zeta)):
            print(1)
        self.xi = temp / self.zeta[:, np.newaxis]
        if np.any(np.isnan(self.xi)):
            print(1)

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        rho = self.caclulate_rho(data)
        return np.argmax(rho, axis=1), rho

    def fit_predict(self, data):

        self.fit(data)
        return self.predict(data)

class VIModel_PY:
    """
    Variational Inference Pitman-Yor process Mixture Models of datas Distributions
    """
    def __init__(self, args):

        self.K = args.K
        self.T = args.T
        self.newJ = self.K
        self.max_k = 700
        self.second_max_iter = args.second_max_iter
        self.args = args
        self.J = 3
        self.N = 300
        self.D = 3
        self.prior = dict()

        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None
        self.pi = None

        self.rho = None
        self.var_theta = None
        self.a = None
        self.b = None
        self.g = None
        self.h = None

        self.temp_top_stick = None
        self.temp_xi_ss = None
        self.temp_k_ss = None

        self.container = {
            'rho': [],
            'var_theta': []
        }

    def init_top_params(self, data):

        self.J = len(data)
        self.D = data[0].shape[1]

        #(self.N, self.D) = data.shape

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        total_data = np.vstack((i for i in data))
        self.prior = {
            'mu': np.sum(total_data, 0) / np.linalg.norm(np.sum(total_data, 0)),
            'zeta': self.args.zeta,
            'u': self.args.u,
            'v': self.args.v,
            'tau': self.args.tau,
            'gamma': self.args.gamma,
            'omega': 0.1, #self.args.omega
            'eta': 0.1, #self.args.eta
        }

        self.u = np.ones(self.K) * self.prior['u']
        self.v = np.ones(self.K) * self.prior['v']
        self.zeta = np.ones(self.K)
        self.xi = np.ones((self.K, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.a = np.ones(self.K)
        self.b = np.ones(self.K)
        self.temp_top_stick = np.zeros(self.K)
        self.temp_xi_ss = np.zeros((self.K, self.D))
        self.temp_k_ss = np.zeros(self.K)

        self.init_update(data)

    def set_temp_zero(self):

        self.temp_top_stick.fill(0.0)
        self.temp_xi_ss.fill(0.0)
        self.temp_k_ss.fill(0.0)

    def init_update(self, x):

        self.var_theta = np.ones((self.T, self.K)) * (1 / self.K)

        for i in range(self.J):
            N = x[i].shape[0]
            self.rho = np.ones((N, self.T)) * (1 / self.T)
            self.temp_top_stick += np.sum(self.var_theta, 0)
            self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
            self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x[i]))

        self.update_zeta_xi()
        self.update_u_v()
        self.update_a_b()

    def calculate_new_com(self):

        threshold = self.args.mix_threshold

        index = np.where(self.pi > threshold)[0]
        self.pi = self.pi[self.pi > threshold]
        self.newJ = self.pi.size

        self.xi = self.xi[index]
        self.k = self.k[index]

        if self.args.verbose:
            print("new component is {}".format(self.newJ))

    def init_second_params(self, N):

        self.rho = np.ones((N, self.T)) * (1 / self.T)

        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_g_h(self.rho)

    def expect_log_sticks(self, a, b, k):

        E_log_1_pi = np.roll(np.cumsum(digamma(b) - digamma(a + b)), 1)
        E_log_1_pi[0] = 0
        return digamma(a) - digamma(a + b) + E_log_1_pi

    def var_inf_2d(self, x, Elogsticks_1nd, ite):

        D = self.D
        Elog_phi = ((x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T)) +
                    (D / 2 - 1) * (digamma(self.u) - np.log(self.v)) -
                    (D / 2 * np.log(2 * np.pi)) -
                    (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) -
                    np.log(iv((D / 2 - 1), self.k) + np.exp(-700)))

        second_max_iter = 5000 if self.second_max_iter == -1 else self.second_max_iter
        self.init_second_params(x.shape[0])
        likelihood = 0.0
        old_likelihood = 1
        converge = 1
        Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)
        for i in range(second_max_iter):
            # compute var_theta

            self.var_theta = self.rho.T.dot(Elog_phi) + Elogsticks_1nd
            log_var_theta, log_n = log_normalize(self.var_theta)
            self.var_theta = np.exp(log_var_theta)

            self.rho = self.var_theta.dot(Elog_phi.T).T + Elogsticks_2nd
            log_rho, log_n = log_normalize(self.rho)
            self.rho = np.exp(log_rho)

            self.update_g_h(self.rho)
            Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)

            likelihood = 0.0
            # compute likelihood
            likelihood += np.sum((Elogsticks_1nd - log_var_theta) * self.var_theta)

            v = np.vstack((self.g, self.h))
            log_alpha = np.log(self.prior['gamma'])
            likelihood += (self.T - 1) * log_alpha
            dig_sum = digamma(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.prior['gamma']])[:, np.newaxis] - v) * (digamma(v) - dig_sum))
            likelihood -= np.sum(gammaln(np.sum(v, 0))) - np.sum(gammaln(v))

            # Z part
            likelihood += np.sum((Elogsticks_2nd - log_rho) * self.rho)

            # X part, the data part
            likelihood += np.sum(self.rho.T * np.dot(self.var_theta, Elog_phi.T))

            if i > 0:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < self.args.threshold:
                break

        self.temp_top_stick += np.sum(self.var_theta, 0)
        self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
        self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x))

        if ite == self.args.max_iter - 1:
            self.container['rho'].append(self.rho)
            self.container['var_theta'].append(self.var_theta)

        return likelihood

    def var_inf(self, x):

        for ite in range(self.args.max_iter):

            self.set_temp_zero()
            Elogsticks_1nd = self.expect_log_sticks(self.a, self.b, self.K)
            for i in range(self.J):
                self.var_inf_2d(x[i], Elogsticks_1nd, ite)

            self.optimal_ordering()
            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi()
            self.update_u_v()
            self.update_a_b()

            if self.args.verbose == 1:
                print('=====> iteration: {}'.format(ite))
            if ite == self.args.max_iter - 1:
                # compute k
                # self.k = self.u / self.v
                # self.k[self.k > self.max_k] = self.max_k
                self.pi = np.exp(self.expect_log_sticks(self.a, self.b, self.K))
                self.calculate_new_com()
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('kappa: {}'.format(self.k))

    def optimal_ordering(self):

        s = [(a, b) for (a, b) in izip(self.temp_top_stick, range(self.K))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        self.temp_top_stick[:] = self.temp_top_stick[idx]
        self.temp_k_ss[:] = self.temp_k_ss[idx]
        self.temp_xi_ss[:] = self.temp_xi_ss[idx]

    def update_u_v(self):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * self.temp_k_ss + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        self.v = self.prior['v'] + self.temp_k_ss * (d_besseli(D / 2 - 1, self.k)) + \
                 self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))

    def update_zeta_xi(self):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + self.temp_xi_ss
        self.zeta = np.linalg.norm(temp, axis=1)
        self.xi = temp / self.zeta[:, np.newaxis]

    def update_g_h(self, rho):

        N_k = np.sum(rho, 0)
        self.g = 1 + N_k - self.prior['eta']
        for i in range(self.T):
            if i == self.T - 1:
                self.h[i] = self.prior['gamma'] + self.T * self.prior['eta']
            else:
                temp = rho[:, i + 1:self.T]
                self.h[i] = self.prior['gamma'] + np.sum(np.sum(temp, 1), 0) + (i+1) * self.prior['eta']

    def update_a_b(self):

        self.a = 1 + self.temp_top_stick - self.prior['omega']
        for i in range(self.K):
            if i == self.K - 1:
                self.b[i] = self.prior['tau'] + self.K * self.prior['omega']
            else:
                temp = self.temp_top_stick[i + 1:self.K]
                self.b[i] = self.prior['tau'] + np.sum(temp) + (i + 1) * self.prior['omega']

    def fit(self, data):

        self.init_top_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        data = np.vstack((i for i in data))
        pred = predict(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        return pred

    def predict_brain(self, data):
        # predict
        data = np.vstack((i for i in data))
        pro = predict_pro(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        pred = np.argmax(pro, axis=1)
        return pred, self.container, pro

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)

class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel_PY(args)
        else:
            pass

    def train(self, data):

        self.model.fit(data)

if __name__ == "__main__":

    #####################basic test#######################
    # data = scio.loadmat('./3d_data.mat')
    # labels = data['z'].reshape(-1)
    # data = data['data']
    #
    # vm = VMFMixture(9, 100)
    # vm.fit(data)
    # pred = vm.predict(data)
    # score = cluster_acc(pred, labels)
    # print("acc: {}".format(score[0]))

    #######################################################

    # data_path = 'USPS_IMBALANCED/train_data.pt'
    # labels_path = 'USPS_IMBALANCED/train_labels.pt'
    # train_data = torch.load(data_path)
    # train_labels = torch.load(labels_path)
    #
    # if isinstance(train_data, torch.Tensor):
    #     train_data = train_data.numpy()
    #
    # if isinstance(train_labels, torch.Tensor):
    #     train_labels = train_labels.numpy()
    #
    # # Apply L2 normalization to the entire dataset
    # normalized_train_data = l2_normalize(train_data)
    # print(np.shape(normalized_train_data))


    #######################balance###############################
    # Splitting the dataset into training and testing sets

    # Train set
    # train_data_final = normalized_train_data[:train_size]
    # train_labels_final = train_labels[:train_size]

    # Test set
    # test_data_final = normalized_train_data[train_size:]
    # test_labels_final = train_labels[train_size:]
    #################################################################

    # train_data_final, test_data_final, train_labels_final, test_labels_final = train_test_split(normalized_train_data,
    #     train_labels, test_size=0.1, random_state=52)
    #
    # vm = VMFMixture(20, 20)
    # vm.fit(train_data_final)
    # print("new_Comonent: {}".format(vm.newJ))
    # pred = vm.predict(test_data_final)
    # score = cluster_acc(pred, test_labels_final)
    # _nmi = NMI(pred, test_labels_final)
    # _ari = ARI(test_labels_final, pred)
    # print("acc: {}".format(score[0]), "nmi: {}".format(_nmi), "ari: {}".format(_ari))

    # for index in range(3):
    #     vm2 = VMFMixture(vm.newJ, 100)
    #     vm2.fit(train_data_final)
    #     print("new_Comonent: {}".format(vm2.newJ))
    #     pred2 = vm2.predict(test_data_final)
    #     score2 = cluster_acc(pred2, test_labels_final)
    #     _nmi2 = NMI(pred2, test_labels_final)
    #     _ari2 = ARI(test_labels_final, pred2)
    #     print("acc: {}".format(score2[0]), "nmi: {}".format(_nmi2), "ari: {}".format(_ari2))
    #     vm = vm2

    # vm = CVMFMixture(80, 100)
    # vm.fit(train_data)
    # # print("new_Comonent: {}".format(vm.newJ))
    # pred = vm.predict(test_data)
    # score = cluster_acc(pred, test_labels)
    # _nmi = NMI(pred, test_labels)
    # _ari = ARI(test_labels, pred)
    # print("acc: {}".format(score[0]), "nmi: {}".format(_nmi), "ari: {}".format(_ari))

    #################################ImageNet-50#################################################################

    # difference datasets config
    # K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim
    # DATA_PARAMS = {
    #     # For the evaluation of simulation data parameters, 500 rounds of iteration can ensure that the
    #     # parameters kappa evaluation is correct, but 10 rounds of iteration can achieve an accuracy of 100%.
    #     'small_data': (10, 5, 0.011, 0, 10, -1, 1e-7, 2, 3),
    #     'adhd': (150, 45, 0.01, 0, 13, -1, 1e-7, 30, 30),
    # }

    parser = argparse.ArgumentParser(prog='HPY-datas',
                                     description='Hierarchical Pitman-Yor process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category', help='choose VIModel_PY:0',
                        default=0, type=int)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='small_data')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1, type=int)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1, type=int)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=80, type=int)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=10, type=int)
    parser.add_argument('-z', '--zeta', dest='zeta', help='zeta', default=0.02, type=float)
    parser.add_argument('-u', '--u', dest='u', help='u', default=0.9, type=float)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01, type=float)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1, type=float)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1, type=float)
    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7, type=float)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01, type=float)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=-1, type=int)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=10,
                        type=int)
    args = parser.parse_args()

    # K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim = DATA_PARAMS[
    #     args.data_name]

    data_path = 'STL10/train_data.pt'
    labels_path = 'STL10/train_labels.pt'
    test_data_path = 'STL10/test_codes.pt'
    test_labels_path = 'STL10/test_labels.pt'
    train_data = torch.load(data_path)
    train_labels = torch.load(labels_path)
    test_data = torch.load(test_data_path)
    test_labels = torch.load(test_labels_path)

    if isinstance(train_data, torch.Tensor):
        train_data = train_data.numpy()
    print("DATA.shape = {}".format(np.shape(train_data)))

    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()

    if isinstance(test_data, torch.Tensor):
        test_data = test_data.numpy()
    print("DATA test.shape = {}".format(np.shape(test_data)))

    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()

    normalized_train_data = l2_normalize(train_data)
    normalized_test_data = l2_normalize(test_data)

    # vm = CVMFMixture(80, 100)
    # vm.fit(train_data)
    # # print("new_Comonent: {}".format(vm.newJ))
    # pred, _ = vm.predict(test_data)
    # score = cluster_acc(pred, test_labels)
    # _nmi = NMI(pred, test_labels)
    # _ari = ARI(test_labels, pred)
    # print("acc: {}".format(score[0]), "nmi: {}".format(_nmi), "ari: {}".format(_ari))

    train_data_new = np.expand_dims(train_data, axis=0)
    test_data_new = np.expand_dims(normalized_test_data, axis=0)

    N_new = (5000 // 500) * 500  # Truncate N to the nearest multiple of 50
    data_truncated = normalized_train_data[:N_new, :]
    # Reshape
    train_data_new_1 = data_truncated.reshape(500, N_new // 500, 512)

    trainer = Trainer(args)
    trainer.train(train_data_new_1)
    pred = trainer.model.predict(test_data_new)
    category = np.unique(np.array(pred))
    console_log(pred, labels=test_labels, model_name='===========hpy-vmf')


#balance:
    #imagenet50:  hpy-vmf acc: 0.6748 nmi: 0.7847 ar: 0.5617 ami: 0.7537

#imbalance:
    #imagenet50:  hpy-vmf acc: 0.7029 nmi: 0.7366 ar: 0.5983 ami: 0.7157
