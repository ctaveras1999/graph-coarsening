import torch
from GraphUtils import noisy_prob, ensure_tensor_bulk, ensure_tensor

def cost_mat(G1: torch.Tensor,
             G2: torch.Tensor,
             P1: torch.Tensor,
             P2: torch.Tensor,
             F1: torch.Tensor,
             F2: torch.Tensor,
             tran: torch.Tensor,
             cost_mode: int = 0,
             alpha: float = 0) -> torch.Tensor:
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
    f1_st = ((G1 ** 2) @ P1).repeat(1, tran.size(1))
    f2_st = (((G2 ** 2) @ P2).T).repeat(tran.size(0), 1)
    cost_st = f1_st + f2_st
    GW_cost = cost_st - 2 * G1 @ tran @ G2.T

    if cost_mode == 0 or (F1 is None or F2 is None):
        cost = GW_cost
    elif cost_mode == 1:
        F1, F2 = ensure_tensor(F1), ensure_tensor(F2)
        tmp1 = ((F1 ** 2) @ torch.ones(F1.size(1), 1)).repeat(1, tran.size(1))
        tmp2 = ((F2 ** 2) @ torch.ones(F2.size(1), 1)).repeat(1, tran.size(0)).T
        tmp3 = 2 * F1 @ F2.T
        W_cost = tmp1 + tmp2 - tmp3
        cost = (1-alpha) * GW_cost + alpha * W_cost
        
    return torch.abs(cost)

def ot_fgw(G1: torch.Tensor,
           G2: torch.Tensor,
           P1: torch.Tensor,
           P2: torch.Tensor,
           F1: torch.Tensor,
           F2: torch.Tensor, 
           ot_layers: int = 30,
           sinkhorn_layers: int = 5,
           alpha: float = 0,
           cost_mode = 0,
           noise_lvl = 7,
           eps=1e-7, 
           sequence=False,
           gamma=5e-2): 
    # initialize transport matrix to product measure by default, otherwise 
    # set initialization manually
    # gamma = 5e-2
    N1, N2 = len(G1), len(G2)
    P1, P2 = noisy_prob(num_nodes=N1, prob=P1, noise_lvl=noise_lvl), noisy_prob(num_nodes=N2, prob=P2, noise_lvl=noise_lvl)
    tran = P1 @ torch.t(P2) # init transport plan

    if sequence:
        tran_seq = torch.zeros(ot_layers, tran.shape[0], tran.shape[1])
        dgw_seq = torch.zeros(ot_layers, 1)

    dual = torch.ones(P1.size()) / P2.size(0)

    Ci = cost_mat(G1, G2, P1, P2, F1, F2, tran, cost_mode, alpha) # cost
    d_gw_last = (Ci * tran).sum()

    best_dist, best_tran = d_gw_last, tran

    for i in range(ot_layers):
        if sequence:
            tran_seq[i] = tran 
            dgw_seq[i] = d_gw_last 

        kernel = torch.exp(-Ci / gamma) * tran
        b = P2 / (torch.t(kernel) @ dual)
        for _ in range(sinkhorn_layers): 
            dual = P1 / (kernel @ b)
            b = P2 / (torch.t(kernel) @ dual)
        tran = torch.abs((dual @ torch.t(b)) * kernel)

        d_gw_curr = (Ci * tran).sum()

        if d_gw_curr < best_dist:
            best_dist, best_tran = d_gw_curr, tran 

        # early termination condition
        gw_diff = torch.norm(d_gw_curr - d_gw_last, 1)
        if gw_diff < eps:
            if sequence:
                for j in range(i, ot_layers):
                    tran_seq[j] = tran
                    dgw_seq[j]  = d_gw_curr
            break

        if not torch.isfinite(d_gw_curr):
            break

        Ci = cost_mat(G1, G2, P1, P2, F1, F2, tran, cost_mode, alpha) # cost
        d_gw_last = d_gw_curr

    if not torch.isfinite(d_gw_curr):
        if 2*gamma < 10:
            print(f"Gamma = {gamma} too small... running again with {2*gamma}\n")
            return ot_fgw(G1, G2, P1, P2, F1, F2, ot_layers, 
                          sinkhorn_layers, alpha,cost_mode, noise_lvl, eps, 
                          sequence, gamma=2*gamma)
        else:
            print(f"Couldn't converge...")
            raise(ValueError)

    d_gw, tran = best_dist, best_tran

    if not sequence:
        return d_gw, tran

    if sequence:
        return d_gw, tran, dgw_seq, tran_seq

def fgwd_from_tran(G1, G2, P1, P2, F1, F2, tran, cost_mode):
    cost = cost_mat(G1, G2, P1, P2, F1, F2, tran, cost_mode=cost_mode)
    return (cost * tran).sum()

def fgwd2(G1, G2, P1=None, P2=None, F1=None, F2=None, alpha=0, ot_layers=30, sinkhorn_layers=5, noise_lvl=8, eps=1e-8, tries=20, verbose=False, sequence=False):
    # if num_ot_layer is not large enough, transport map may not have converged 
    if verbose:
        print("-"*50 + "Computing FGWD" + "-"*50)

    G1, G2 = ensure_tensor_bulk([G1, G2])
    cost_mode = int((alpha != 0) and not (F1 is None or F2 is None))
    best_dist, best_tran = float("inf"), None
    dists = []
    if sequence:
        N1, N2 = G1.shape[0], G2.shape[0]
        d_gw_seqs, tran_seqs = torch.zeros(tries, ot_layers), torch.zeros(tries, ot_layers, N1, N2)

    for iter in range(tries):
        res = ot_fgw(G1, G2, P1, P2, F1, F2, ot_layers, sinkhorn_layers, alpha, cost_mode, noise_lvl, eps, sequence)
        if not sequence:
            d_fgw, tran = res
        else:
            d_fgw, tran, d_gw_seq_i, tran_seq_i = res
            d_gw_seqs[iter] = d_gw_seq_i.reshape(-1)
            tran_seqs[iter] = tran_seq_i

        dists.append(d_fgw)
        if (d_fgw != torch.nan) and (d_fgw < best_dist):
            best_dist, best_tran = d_fgw, tran

    if verbose:
        import numpy as np
        print(f"avg dist = {np.round(np.mean(dists), 3)}")
        print(f"std dist = {np.round(np.std(dists), 3)}")
        print("-"*50 + "Done!" + "-"*50)

    if not sequence:
        return best_dist, best_tran
    
    else:
        return best_dist, best_tran, d_gw_seqs, tran_seqs

def fgwd(M1, M2, alpha=0, ot_layers=30, sinkhorn_layers=5, noise_lvl=8, 
         eps=1e-8, tries=20, verbose=False, sequence=False):
    
    G1, P1, F1 = M1.get_graph(), M1.get_prob(), M1.get_feat()
    G2, P2, F2 = M2.get_graph(), M2.get_prob(), M2.get_feat()

    return fgwd2(G1=G1, G2=G2, P1=P1, P2=P2, F1=F1, F2=F2, alpha=alpha, 
                 ot_layers=ot_layers, sinkhorn_layers=sinkhorn_layers, 
                 noise_lvl=noise_lvl, eps=eps, tries=tries, verbose=verbose,
                 sequence=sequence)

