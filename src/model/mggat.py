"""
MG-GAT: Multi-Graph Graph Attention Network

Fix: omega weights correctly applied per-graph [Eq.3]
  - Three biz graphs each compute attention scores independently
  - omega_g weights the scores before softmax

Paper: Leng, Liu, Ruiz -- SSRN 3696092
"""

import torch
import torch.nn as nn


class MGGAT(nn.Module):
    def __init__(
        self,
        su: int,
        sb: int,
        n_users: int,
        n_biz: int,
        n_biz_graphs: int = 3,
        d0_u: int = 64,
        d0_b: int = 64,
        d1_u: int = 128,
        d1_b: int = 128,
        kf: int = 64,
        actv1: str = 'elu',
        actv2: str = 'relu',
        r_min: float = 1.0,
        r_max: float = 5.0,
    ):
        super().__init__()
        self.kf = kf
        self.r_min = r_min
        self.r_max = r_max
        self.d0_u = d0_u
        self.d0_b = d0_b
        self.n_biz_graphs = n_biz_graphs

        # Layer 1: Linear, no bias, no activation [Eq.2]
        self.W1_u = nn.Linear(su, d0_u, bias=False)
        self.W1_b = nn.Linear(sb, d0_b, bias=False)

        # Layer 2: Attention vectors [Eq.3]
        self.a_u = nn.Parameter(torch.empty(2 * d0_u))
        self.a_b = nn.Parameter(torch.empty(2 * d0_b))
        nn.init.xavier_uniform_(self.a_u.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_b.unsqueeze(0))

        # Multi-graph learnable weights omega_g [Eq.3]
        self.omega = nn.Parameter(torch.ones(n_biz_graphs) / n_biz_graphs)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Layer 4: Non-linear [Eq.5]
        self.W2_u  = nn.Linear(d0_u, d1_u, bias=False)
        self.W2_us = nn.Linear(su,   d1_u, bias=False)
        self.b1_u  = nn.Parameter(torch.zeros(d1_u))
        self.W2_b  = nn.Linear(d0_b, d1_b, bias=False)
        self.W2_bs = nn.Linear(sb,   d1_b, bias=False)
        self.b1_b  = nn.Parameter(torch.zeros(d1_b))

        # Layer 5: Final [Eq.6]
        self.W3_u = nn.Linear(d1_u, kf, bias=False)
        self.W3_b = nn.Linear(d1_b, kf, bias=False)

        # H4: graph reg only on these [Section 3.4]
        self.H4_u = nn.Embedding(n_users, kf)
        self.H4_b = nn.Embedding(n_biz,   kf)
        nn.init.normal_(self.H4_u.weight, std=0.01)
        nn.init.normal_(self.H4_b.weight, std=0.01)

        # Prediction bias [Eq.7]
        self.bias_u      = nn.Embedding(n_users, 1)
        self.bias_b      = nn.Embedding(n_biz,   1)
        self.bias_global = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.bias_u.weight)
        nn.init.zeros_(self.bias_b.weight)

        self.actv1 = self._get_activation(actv1)
        self.actv2 = self._get_activation(actv2)

    def _get_activation(self, name: str):
        return {
            'elu': nn.ELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(),
            'selu': nn.SELU(), 'sigmoid': nn.Sigmoid(), 'linear': nn.Identity(),
        }[name]

    # FR (Feature Relevance) [Definition 2]
    @property
    def FR_user_self(self):
        return (self.a_u[:self.d0_u] @ self.W1_u.weight).detach()

    @property
    def FR_user_nb(self):
        return (self.a_u[self.d0_u:] @ self.W1_u.weight).detach()

    @property
    def FR_biz_self(self):
        return (self.a_b[:self.d0_b] @ self.W1_b.weight).detach()

    @property
    def FR_biz_nb(self):
        # FR = ∑_g ω_g · a_{b,nb}^T W_b = (∑_g ω_g) · a_{b,nb}^T W_b
        # softmax(omega).sum() == 1 always, so omega cancels out [Definition 2]
        return (self.a_b[self.d0_b:] @ self.W1_b.weight).detach()

    def _softmax_by_dst(self, e, dst, N):
        e_max = torch.full((N,), float('-inf'), device=e.device)
        e_max.scatter_reduce_(0, dst, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, device=e.device).scatter_add_(0, dst, e_exp)
        return e_exp / (e_sum[dst] + 1e-16)

    def _compute_nig_user(self, H1_u, edge_index_u):
        src, dst = edge_index_u[0], edge_index_u[1]
        h_cat = torch.cat([H1_u[dst], H1_u[src]], dim=1)
        e = self.leaky_relu((h_cat * self.a_u).sum(dim=1))
        return self._softmax_by_dst(e, dst, H1_u.shape[0])

    def _compute_nig_biz(self, H1_b, edge_indices_b):
        """
        Multi-graph attention [Eq.3]:
          α_b^(l→j) = softmax_j(LeakyReLU(∑_{g∈B} ω_g · a_b^T [H_j || H_l]))
        Same (i,j) pair may appear in multiple graphs — their weighted scores
        must be summed before softmax, not treated as separate edges.
        """
        omega = torch.softmax(self.omega, dim=0)
        N = H1_b.shape[0]

        all_src, all_dst, all_e = [], [], []
        for g, edge_index in enumerate(edge_indices_b):
            src, dst = edge_index[0], edge_index[1]
            h_cat = torch.cat([H1_b[dst], H1_b[src]], dim=1)
            e_g = omega[g] * (h_cat * self.a_b).sum(dim=1)
            all_src.append(src)
            all_dst.append(dst)
            all_e.append(e_g)

        all_src = torch.cat(all_src)
        all_dst = torch.cat(all_dst)
        all_e   = torch.cat(all_e)

        # Merge duplicate (i,j) pairs: sum weighted scores across graphs [Eq.3 ∑_g]
        keys = all_src * N + all_dst
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        n_unique = unique_keys.shape[0]
        merged_e = torch.zeros(n_unique, device=H1_b.device)
        merged_e.scatter_add_(0, inverse, all_e)

        merged_src = unique_keys // N
        merged_dst = unique_keys  % N

        merged_e = self.leaky_relu(merged_e)
        alpha = self._softmax_by_dst(merged_e, merged_dst, N)
        return merged_src, merged_dst, alpha

    def _aggregate(self, H1, alpha, src, dst, N):
        weighted = H1[src] * alpha.unsqueeze(1)
        H2 = torch.zeros(N, H1.shape[1], device=H1.device)
        H2.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted), weighted)
        return H2

    def forward(
        self,
        S_u: torch.Tensor,
        S_b: torch.Tensor,
        edge_index_u: torch.Tensor,
        edge_indices_b: list,        # [(2,E_geo), (2,E_cov), (2,E_cat)]
        user_idx: torch.Tensor,
        biz_idx:  torch.Tensor,
    ):
        n_u = S_u.shape[0]
        n_b = S_b.shape[0]

        # Layer 1 [Eq.2]
        H1_u = self.W1_u(S_u)
        H1_b = self.W1_b(S_b)

        # Layer 2: NIG [Eq.3]
        alpha_u = self._compute_nig_user(H1_u, edge_index_u)
        src_b, dst_b, alpha_b = self._compute_nig_biz(H1_b, edge_indices_b)

        # Layer 3: Aggregation [Eq.4]
        src_u, dst_u = edge_index_u[0], edge_index_u[1]
        H2_u = self._aggregate(H1_u, alpha_u, src_u, dst_u, n_u)
        H2_b = self._aggregate(H1_b, alpha_b, src_b, dst_b, n_b)

        # Layer 4 [Eq.5]
        H3_u = self.actv1(self.W2_u(H2_u) + self.W2_us(S_u) + self.b1_u)
        H3_b = self.actv1(self.W2_b(H2_b) + self.W2_bs(S_b) + self.b1_b)

        # Layer 5 [Eq.6]
        U_all = self.actv2(self.W3_u(H3_u)) + self.H4_u.weight
        B_all = self.actv2(self.W3_b(H3_b)) + self.H4_b.weight

        # Prediction [Eq.7]
        U_q = U_all[user_idx]
        B_q = B_all[biz_idx]
        logit = (U_q * B_q).sum(dim=1) \
              + self.bias_u(user_idx).squeeze(1) \
              + self.bias_b(biz_idx).squeeze(1) \
              + self.bias_global
        pred = (self.r_max - self.r_min) * torch.sigmoid(logit) + self.r_min
        return pred, U_all, B_all


def graph_laplacian_reg(H4, edge_index, theta2):
    """Tr(H4^T * L_tilde * H4) [Eq.8] -- only on H4 [Section 3.4]"""
    src, dst = edge_index[0], edge_index[1]
    diff = H4[src] - H4[dst]
    return 0.5 * (diff ** 2).sum() + theta2 * (H4 ** 2).sum()
