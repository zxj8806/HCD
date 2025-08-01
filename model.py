import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

from preprocessing import sparse_to_tuple, preprocess_graph
from cluster_utils import (
    Clustering_Metrics,
    GraphConvSparse,
    purity_score,
    ClusterAssignment,
)

from vmfmix.vmf import VMFMixture
from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform
from torch.distributions.kl import kl_divergence


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2, device=None):
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def compute_diffusion_params(timesteps: int, device):
    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m = torch.sqrt(1.0 - alphas_cumprod)
    rev_cum = torch.flip(sqrt_1m, dims=[0]).cumsum(0)
    cum_sqrt_1m = torch.flip(rev_cum, dims=[0])
    return sqrt_1m, cum_sqrt_1m


class HCD(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.num_neurons = kwargs["num_neurons"]
        self.num_features = kwargs["num_features"]
        self.embedding_size = kwargs["embedding_size"]
        self.nClusters = kwargs["nClusters"]
        self.T = kwargs.get("T", 30)

        act = {
            "ReLU": F.relu,
            "Sigmoid": F.sigmoid,
            "Tanh": F.tanh,
        }.get(kwargs.get("activation", "ReLU"), F.relu)
        self.activation = act

        self.kl_hyp_weight = kwargs.get("kl_hyp_weight", 0.1)
        init_kappa = kwargs.get("init_kappa", 10.0)
        self.align_weight = kwargs.get("align_weight", 0.1)
        self.align_alpha = kwargs.get("align_alpha", 2)
        self.align_num_neg = kwargs.get("align_num_neg", 1)
        self.align_margin = kwargs.get("align_margin", 1.0)
        self.cluster_reg_weight = kwargs.get("cluster_reg_weight", 0.1)
        self.entropy_reg_weight = kwargs.get("entropy_reg_weight", 2e-3)

        self.log_kappa = nn.Parameter(torch.tensor(init_kappa))
        self.z_weight = nn.Parameter(torch.zeros(self.T))  # time weights

        self.base_gcn = GraphConvSparse(
            self.num_features, self.num_neurons, self.activation
        )
        self.gcn_mean = GraphConvSparse(
            self.num_neurons, self.embedding_size, lambda x: x
        )
        self.gcn_logsigma2 = GraphConvSparse(
            self.num_neurons, self.embedding_size, lambda x: x
        )

        self.assignment = ClusterAssignment(
            self.nClusters, self.embedding_size, kwargs["alpha"]
        )

    def kappa(self):
        return F.softplus(self.log_kappa)

    def aggregate_poe(self, mus, logs2):
        T = len(mus)
        w_t = F.softmax(self.z_weight[:T], 0).view(T, 1, 1)  # (T,1,1)
        mu_stack = torch.stack(mus, 0)
        logvar_stack = torch.stack(logs2, 0)
        precision = torch.exp(-logvar_stack)
        prec_w = w_t * precision
        return (prec_w * mu_stack).sum(0) / prec_w.sum(0)

    @staticmethod
    def _sample_negative_pairs(adj_dense, total_neg, device):
        n = adj_dense.size(0)
        neg_r = torch.empty(0, dtype=torch.long, device=device)
        neg_c = torch.empty(0, dtype=torch.long, device=device)
        invalid = adj_dense.bool() | torch.eye(n, device=device, dtype=torch.bool)
        while neg_r.numel() < total_neg:
            r = torch.randint(0, n, (total_neg,), device=device)
            c = torch.randint(0, n, (total_neg,), device=device)
            ok = ~invalid[r, c]
            neg_r = torch.cat([neg_r, r[ok]])
            neg_c = torch.cat([neg_c, c[ok]])
        return neg_r[:total_neg], neg_c[:total_neg]

    @staticmethod
    def align_loss(z, adj, alpha=2, num_neg=1, margin=1.0):
        if adj.is_sparse:
            row, col = adj.coalesce().indices()
        else:
            row, col = adj.nonzero(as_tuple=True)
        diff_pos = z[row] - z[col]
        pos = diff_pos.norm(2, 1).pow(alpha).mean()

        total_neg = row.size(0) * num_neg
        adj_d = adj.to_dense() if adj.is_sparse else adj.clone()
        nr, nc = HCD._sample_negative_pairs(adj_d, total_neg, z.device)
        diff_neg = z[nr] - z[nc]
        neg = F.relu(margin - diff_neg.norm(2, 1)).pow(alpha).mean()
        return pos + neg

    def encode(self, x, adj, T=1):
        mus, logs2, zs = [], [], []
        for _ in range(T):
            h = self.base_gcn(x, adj)
            mu = self.gcn_mean(h, adj)
            ls2 = self.gcn_logsigma2(h, adj)
            z = torch.randn_like(mu) * torch.exp(ls2 / 2) + mu
            mus.append(mu)
            logs2.append(ls2)
            zs.append(z)
        return mus, logs2, zs

    def decode_diffusion(self, zs, start=1):
        T = len(zs)
        sqrt_1m, cum = compute_diffusion_params(T, zs[0].device)
        norm = cum[start - 1]
        acc = None
        for tau, z_tau in enumerate(zs[start - 1 :], start=start):
            z_tau = F.normalize(z_tau, 2, 1)
            sim = z_tau @ z_tau.t()
            w = sqrt_1m[tau - 1]
            acc = sim * w if acc is None else acc + sim * w
        return torch.clamp(acc / norm, 0, 1)

    def train(
        self,
        features,
        adj_norm,
        adj_label,
        y,
        weight_tensor,
        norm,
        optimizer="Adam",
        epochs=1000,
        lr=5e-3,
        kappa_lr=1e-3,
        save_path="./results/",
        dataset="Cora",
        run_id: str = None,
    ):
        import os, time, json, random
        import numpy as np
        from tqdm import tqdm

        def _cpu_byte(t):
            if isinstance(t, torch.ByteTensor) and t.device.type == "cpu":
                return t
            return torch.tensor(t, dtype=torch.uint8, device="cpu")

        def get_rng_state():
            return {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().clone(),
                "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            }

        def set_rng_state(state):
            try:
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
            except Exception:
                pass
            try:
                torch.set_rng_state(_cpu_byte(state["torch"]))
            except Exception:
                pass
            if torch.cuda.is_available() and state.get("torch_cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all([_cpu_byte(s) for s in state["torch_cuda"]])
                except Exception:
                    pass

        os.makedirs(save_path, exist_ok=True)
        base_dir = os.path.join(save_path, dataset)
        os.makedirs(base_dir, exist_ok=True)

        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            need_resume = False
        else:
            need_resume = os.path.isfile(os.path.join(base_dir, run_id, "epoch0.ckpt"))

        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        ckpt0_path = os.path.join(run_dir, "epoch0.ckpt")

        base_params = [p for n, p in self.named_parameters() if n != "log_kappa"]
        optim_cls = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD
        opt = optim_cls(
            [
                {"params": base_params, "lr": lr},
                {"params": [self.log_kappa], "lr": kappa_lr},
            ],
            **({"momentum": 0.9} if optimizer == "SGD" else {}),
        )

        device = getattr(features, "device", next(self.parameters()).device)
        b_acc = 0.0

        if need_resume:
            ckpt = torch.load(ckpt0_path, map_location=device, weights_only=False)
            self.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optim_state"])
            if "rng_state" in ckpt:
                set_rng_state(ckpt["rng_state"])
            meta = ckpt.get("meta", {})
            b_acc = meta.get("b_acc", 0.0)
        else:
            rng_state = get_rng_state()
            meta = {
                "run_id": run_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "epochs": epochs,
                "optimizer": optimizer,
                "lr": lr,
                "kappa_lr": kappa_lr,
                "dataset": dataset,
                "b_acc": b_acc,
                "model_class": self.__class__.__name__,
            }
            torch.save(
                {
                    "epoch": 0,
                    "model_state": self.state_dict(),
                    "optim_state": opt.state_dict(),
                    "rng_state": rng_state,
                    "meta": meta,
                },
                ckpt0_path,
            )
            with open(os.path.join(run_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[Init] epoch0.ckpt -> {ckpt0_path}")

        vmf = VMFMixture(n_cluster=self.nClusters, max_iter=100)
        bar = tqdm(range(epochs), desc="training")

        for ep in bar:
            opt.zero_grad()

            mus, logs2, zs = self.encode(features, adj_norm, T=self.T)
            z = self.aggregate_poe(mus, logs2)

            adj_out = self.decode_diffusion(zs)
            loss_rec = norm * torch.nn.functional.binary_cross_entropy(
                adj_out.reshape(-1),
                adj_label.to_dense().reshape(-1),
                weight=weight_tensor,
            )

            z_u = torch.nn.functional.normalize(z, 2, 1)
            kappa_b = torch.full((z_u.size(0), 1), self.kappa().item(), device=z.device)
            qz = VonMisesFisher(z_u, kappa_b)
            pz = HypersphericalUniform(z_u.size(1) - 1)
            loss_kl = kl_divergence(qz, pz).mean()

            loss_aln = self.align_loss(
                z_u, adj_label, self.align_alpha, self.align_num_neg, self.align_margin
            )

            p = self.assignment(z_u)
            centers = torch.nn.functional.normalize(self.assignment.cluster_centers, 2, 1)
            intra = ((p[:, :, None] * (z_u[:, None, :] - centers[None, :, :]).pow(2))).sum() / z_u.size(0)
            inter = torch.pdist(centers, 2).mean()
            loss_clu = intra / (inter + 1e-9)

            loss_ent = (p * torch.log(p + 1e-9)).sum() / p.size(0)

            loss = (
                loss_rec
                + self.kl_hyp_weight * loss_kl
                + self.align_weight * loss_aln
                + self.cluster_reg_weight * loss_clu
                + self.entropy_reg_weight * loss_ent
            )
            loss.backward()
            opt.step()

            vmf.fit(z_u.detach().cpu().numpy())
            y_pred = vmf.labels_ if hasattr(vmf, "labels_") else vmf.predict(z_u.detach().cpu().numpy())
            acc = Clustering_Metrics(y, y_pred).evaluationClusterModelFromLabel()[0]

            bar.set_description(
                f"loss={loss.item():.4f} rec={loss_rec.item():.4f} "
                f"kl={loss_kl.item():.4f} aln={loss_aln.item():.4f} "
                f"clu={loss_clu.item():.4f} ent={loss_ent.item():.4f} "
            )

            if acc > b_acc:
                b_acc = acc
                meta["b_acc"] = b_acc
                if vmf.xi.shape[0] == self.nClusters:
                    self.assignment.cluster_centers.data = torch.tensor(
                        vmf.xi, dtype=torch.float, device=z.device
                    )
                torch.save(self.state_dict(), os.path.join(run_dir, "b_acc.pk"))


        print("ACC:", b_acc)
        return y_pred, y


