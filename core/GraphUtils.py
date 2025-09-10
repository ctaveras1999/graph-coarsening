import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

Gs_to_As = lambda Gs : [nx.to_numpy_array(G) for G in Gs]
As_to_Ls = lambda As : [np.diag(A @ np.ones(A.shape[0])) - A for A in As]

G_to_A = lambda G : nx.to_numpy_array(G)
A_to_L = lambda A : np.diag(A @ np.ones(A.shape[0])) - A
A_to_G = lambda A : nx.from_numpy_array(A)
L_to_A = lambda L : np.diag(np.diag(L)) - L
G_to_L = lambda G : A_to_L(G_to_A(G))

def noisy_prob(num_nodes=None, prob=None, noise_lvl=8):
    assert((not num_nodes is None) or (not prob is None))

    if prob is not None:
        P = ensure_tensor(prob)
    else:
        P = torch.ones(num_nodes,1) / num_nodes
    
    noise = (2 * torch.rand(num_nodes, 1) - 1) / 10**noise_lvl
    P = P + noise
    P = P / torch.norm(P, 1)
    return P

def noisy_prob_pair(N1, N2, noise_lvl=8):
    if N1 == N2:
        P1 = P2 = noisy_prob(N1, None, noise_lvl)
    else:
        P1, P2 = noisy_prob(N1, None, noise_lvl), noisy_prob(N2, None, noise_lvl)
    return P1, P2

def rand_simplex(k):
    return tuple(np.random.dirichlet((1,)*k))

def simplex_sample(N, d):
    return np.array([rand_simplex(d) for i in range(N)]).T

def rand_simplex(k):
    return tuple(np.random.dirichlet((1,)*k))

def simplex_sample(N, d):
    return np.array([rand_simplex(d) for i in range(N)]).T

def ensure_tensor(G, requires_grad=False):
    G_tensor = None
    if G is None:
        return G
    elif isinstance(G, nx.Graph):
        G_tensor = torch.tensor(G_to_A(G), dtype=torch.float32)
    elif isinstance(G, np.ndarray):
        G_tensor = torch.tensor(G, dtype=torch.float32)
    elif isinstance(G, list):
        G_tensor = torch.tensor(G, dtype=torch.float32)
    else:
        G_tensor = G.to(dtype=torch.float32)
    if len(G_tensor.shape) == 1:
        G_tensor = G_tensor.reshape(-1,1)
    
    if requires_grad:
        G_tensor.requires_grad_()

    return G_tensor

def ensure_tensor_bulk(Gs):
    res = [ensure_tensor(G) for G in Gs]
    return res
