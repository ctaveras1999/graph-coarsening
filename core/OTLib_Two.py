import networkx as nx
import torch
import numpy as np

def cost_mat(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             p_s: torch.Tensor,
             p_t: torch.Tensor,
             tran: torch.Tensor,
             emb_s: torch.Tensor = None,
             emb_t: torch.Tensor = None,
             cost_mode: int = 0,
             alpha: float = 0.5) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
        emb_s: (ns, d) matrix
        emb_t: (nt, d) matrix
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = ((cost_s ** 2) @ p_s).repeat(1, tran.size(1))
    f2_st = (((cost_t ** 2) @ p_t).T).repeat(tran.size(0), 1)
    cost_st = f1_st + f2_st
    FGW_cost = cost_st - 2 * cost_s @ tran @ cost_t.T

    cost = FGW_cost
    if cost_mode == 0 or (emb_s is None or emb_t is None):
        cost = FGW_cost
    else:
        if cost_mode == 1:
            tmp1 = emb_s @ emb_t.T
            tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
            tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
            W_cost = (1 - tmp1 / (tmp2 @ tmp3.T))
        else:
            tmp1 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, tran.size(1))
            tmp2 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, tran.size(0)).T
            tmp3 = 2 * emb_s @ emb_t.T
            W_cost = tmp1 + tmp2 - tmp3
        cost = (1-alpha) * FGW_cost + alpha * W_cost
        
    return cost

def ot_fgw(cost_s: torch.Tensor,
           cost_t: torch.Tensor,
           p_s: torch.Tensor,
           p_t: torch.Tensor,
           ot_method: str,
           gamma: float,
           num_layer: int,
           num_sinkhorn: int = 5,
           emb_s: torch.Tensor = None,
           emb_t: torch.Tensor = None, 
           alpha: float = 0.5, 
           cost_mode = 0,
           trans_mode = 0):
    # initialize transport matrix to product measure by default, otherwise 
    # set initialization manually
    # tran = tran_init if tran_init is not None else p_s @ torch.t(p_t) 
    if trans_mode == 1:
        assert(cost_s.shape == cost_t.shape)
        N = cost_s.shape[0] 
    p_s, p_t = torch.Tensor(p_s).reshape(-1, 1), torch.Tensor(p_t).reshape(-1, 1)
    tran = p_s @ torch.t(p_t) if trans_mode == 0 else torch.eye(N)/N
    if ot_method == 'ppa':
        dual = torch.ones(p_s.size()) / p_s.size(0)
        for m in range(num_layer):
            cost = cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, cost_mode, alpha)
            # cost /= torch.max(cost) 
            kernel = torch.exp(-cost / gamma) * tran
            b = p_t / (torch.t(kernel) @ dual)
            for i in range(num_sinkhorn): 
                dual = p_s / (kernel @ b)
                b = p_t / (torch.t(kernel) @ dual)
            tran = (dual @ torch.t(b)) * kernel

    elif ot_method == 'b-admm':
        all1_s = torch.ones(p_s.size())
        all1_t = torch.ones(p_t.size())
        dual = torch.zeros(p_s.size(0), p_t.size(0))
        for m in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a
            dual = dual + gamma * (tran - aux)

            cost = cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t, cost_mode, alpha)
            # cost /= torch.max(cost)
            kernel_t = torch.exp(-(cost + dual) / gamma) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, cost_mode, alpha) * tran).sum()
    return d_gw, tran

def noisy_unif(num_nodes, M=1e8):
    x = np.abs((2 * np.random.random((num_nodes,1)) - 1)/M + 1/num_nodes)
    return torch.Tensor(x / np.linalg.norm(x, 1))

def gwd(G1, G2, prob1=None, prob2=None, num_ot_layers=30, num_sinkhorn=5, gamma=5e-2, ot_method='ppa', unif_noise=1e7):
    Gs = [G1, G2]
    for i, G in enumerate(Gs):
        if isinstance(G, nx.Graph):
            Gs[i] = torch.Tensor(nx.to_numpy_array(G))
        elif isinstance(G, np.ndarray):
            Gs[i] = torch.Tensor(G)
    G1, G2 = Gs
    N1, N2 = G1.shape[0], G2.shape[1]
    if prob1 is None and prob2 is None:
        if N1 == N2:
            prob1 = prob2 = noisy_unif(N1, unif_noise)
        else:
            prob1, prob2 = noisy_unif(N1, unif_noise), noisy_unif(N2, unif_noise)
    elif prob1 is None:
        prob1 = noisy_unif(N1, unif_noise)
    elif prob2 is None:
        prob2 = noisy_unif(N2, unif_noise)

    N1 = G1.shape[0]
    N2 = G2.shape[0]

    d_gw, tran = ot_fgw(G1, G2, prob1, prob2, ot_method, gamma, num_ot_layers, num_sinkhorn, None, None, 0, 0)
   
    return d_gw, tran