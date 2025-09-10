import sys
sys.path.insert(1, '../GWGraphCoarsening')

import torch
from GraphUtils import ensure_tensor, A_to_G
from AlgOT import fgwd, cost_mat
import networkx as nx
# import GWGraphCoarsening.coarsening as coarsening 
import numpy as np
import measure
# import CoarsenUtil as util
from CoarsenUtil import *
from sklearn.cluster import SpectralClustering
from pygkernels.cluster import KKMeans
from sklearn.cluster import KMeans
import laplacian 
import copy 

def true_trans(Q, mu=None):
    if mu is None:
        n = Q.shape[0]
        mu = torch.ones((n))/n
    Cp = (Q > 0).to(torch.float32).clone().detach()
    Q2 = torch.diag(mu)@Cp
    mu, mu_prime = torch.sum(Cp, 1)/Cp.shape[1], torch.sum(Cp, 0)/Cp.shape[0]
    # print(torch.diag(torch.ones(Cp.shape[1])/Cp.shape[1]))
    # print(Cp.shape)
    return Q2

def pair_distortion(x, y, i, j, mu):
    idxs = torch.tensor([i,j])
    mask = torch.ones(x.numel(), dtype=bool)
    mask[idxs] = False
    x1, y1, mu1 = x[mask], y[mask], mu[mask]
    xi, xj, yi, yj = x[i], x[j], y[i], y[j]
    mu_bar, mu_hat = (mu[i] + mu[j]) / 2, torch.sqrt(mu[i] * mu[j])
    multi = mu_hat ** 2 / (2 * mu_bar**2)
    term1 = 2 * mu_bar * torch.sum(mu1 * (x1 - y1) ** 2)
    term2 = (mu[i] * (xi - yi))**2 + (mu[j] * (xj - yj))**2 
    term3 = (mu_hat * (xi - yj))**2 / 2
    err = multi * (term1 + term2 + term3)
    return err

def sorted_heuristic(measure_net, means_type=False):
    graph = nx.from_numpy_array(measure_net.get_graph().numpy())
    A = measure_net.graph
    N = A.shape[0]

    common_neighbors = torch.zeros(measure_net.num_nodes, measure_net.num_nodes) # shouldn't have to recompute every iteration.    
    
    if means_type == 0:
        nbhds = [set(graph.neighbors(v)) for v in np.sort(graph.nodes())]           # only update effected nodes and add super node.
                                                                                    # supernode is union of node neighborhoods. 
                                                                                    # all nodes adjacent to super node are now 
                                                                                    # no longer connected to original nodes, but 
                                                                                    # are connected to super node

    for vi in range(N):
        for vj in range(N):
            if vj <= vi:
                # print(f"rejected {vi, vj}")
                continue

            if means_type == 0 :
                Ni, Nj = nbhds[vi], nbhds[vj]

                if not isinstance(Ni, set):
                    Ni = set([Ni])
                
                if not isinstance(Nj, set):
                    Nj = set([Nj]) 

                all_neigh = list(Ni.union(Nj))

                jaccard_numer, jaccard_denom = 0, 0 
                for k, vk in enumerate(all_neigh):
                    alpha = 1/2 if ((vi == vk) or (vj == vk)) else 1.0
                    numer, denom = np.sort([A[vi][vk], A[vj][vk]])
                    jaccard_numer += alpha * numer
                    jaccard_denom += alpha * denom

                if jaccard_denom == 0:
                    jaccard = 0
                    common_neighbors[vi, vj] = common_neighbors[vj,vi] = 0

                else:
                    jaccard = float(jaccard_numer / jaccard_denom)            
                    common_neighbors[vi,vj] = common_neighbors[vj,vi] = jaccard
            else:
                M, mu = measure_net.graph, measure_net.prob
                x, y = M[vi], M[vj]
                common_neighbors[vi, vj] = common_neighbors[vj, vi] = pair_distortion(x, y, vi, vj, mu)

    a = torch.triu_indices(*common_neighbors.shape, offset=1)
    common_neighbors = torch.Tensor([common_neighbors[u][v] for u,v in zip(a[0], a[1])])# if u != v])
    common_neighbors_sorted = torch.sort(common_neighbors, descending=False).values
    common_neighbors_sorted_idx = torch.argsort(common_neighbors, descending=False)
    common_neighbors_sorted_idx = torch.vstack([a[:,i] for i in common_neighbors_sorted_idx])

    return common_neighbors_sorted, common_neighbors_sorted_idx

def extract_nodes(mat, N, vi, vf):
    M_prime = torch.zeros(N - 1, N - 1)

    starts_in_cand, ends_in_cand   = [0,  vi+1, vf+1], [vi,  vf, N] 
    starts_in, ends_in = [], []
    starts_out, ends_out = [0], []

    for x,y in zip(starts_in_cand, ends_in_cand):
        if x == y:
            continue
        if x > N:
            continue

        starts_in.append(x)
        ends_in.append(y)

        ends_out.append(starts_out[-1] + y - x)

        if starts_out[-1] + y - x < N - 2:
            starts_out.append(starts_out[-1]+y-x)

    for i in range(len(starts_in) ** 2):
        row_idx, col_idx = i // len(starts_in), i % len(starts_in)

        t1, b1 = starts_in[row_idx], ends_in[row_idx]
        l1, r1 = starts_in[col_idx], ends_in[col_idx]

        t2, b2 = starts_out[row_idx], ends_out[row_idx]
        l2, r2 = starts_out[col_idx], ends_out[col_idx]

        M_prime[t2:b2,l2:r2] = mat[t1:b1,l1:r1]
    
    return M_prime

def extract_from_mat(measure_network, vi, vf):
    mat = measure_network.get_graph()
    N = mat.shape[0]
    vi, vf = int(torch.min(torch.tensor([vi, vf]))), int(torch.max(torch.tensor([vi, vf])))
    prob = measure_network.get_prob()

    M_prime = extract_nodes(mat, N, vi, vf) 

    m1_horz = torch.hstack([mat[vi,:vi], mat[vi,vi+1:vf], mat[vi,vf+1:]]).reshape(1,-1)
    m2_horz = torch.hstack([mat[vf,:vi], mat[vf,vi+1:vf], mat[vf,vf+1:]]).reshape(1,-1)
    m1 = torch.vstack([m1_horz, m2_horz])

    m1_vert = torch.hstack([mat[:vi,vi], mat[vi+1:vf,vi], mat[vf+1:,vi]]).reshape(-1,1)
    m2_vert = torch.hstack([mat[:vi,vf], mat[vi+1:vf,vf], mat[vf+1:,vf]]).reshape(-1,1)
    m2 = torch.hstack([m1_vert, m2_vert])
    m3 = torch.tensor([mat[vi, vi], mat[vi, vf], mat[vf, vi], mat[vf, vf]]).reshape(2,2)

    pi, pf = prob[vi], prob[vf]
    theta = pi / (pi + pf)
    m1_merge = theta * m1[0,:] + (1-theta) * m1[1,:]
    m2_merge = theta * m2[:,0] + (1-theta) * m2[:,1]
    m3_merge = (pi**2 * m3[0,0] + pi*pf* (m3[0,1] + m3[1,0]) + pf**2 * m3[1,1])
    m3_merge = m3_merge / (pi + pf)**2

    M_prime[-1,:-1] = m1_merge
    M_prime[:-1,-1] = m2_merge
    M_prime[-1,-1]  = m3_merge

    new_prob = torch.hstack([prob[:vi,0], prob[vi+1:vf,0], prob[vf+1:,0], prob[vi,0] + prob[vf,0]]).reshape(prob.shape[0]-1, 1)
    M_prime = MeasureNetwork(M_prime, new_prob, None) # update features 

    return M_prime

def get_trans(m_net, vi, vf):
    trans = [None] * m_net.num_nodes
    trans[vi] = (-1, m_net.prob[vi])
    trans[vf] = (-1, m_net.prob[vf])
    columns = [i for i in range(m_net.num_nodes) if i not in [vi, vf]]
    for j, k in enumerate(columns):
        trans[k] = (j, m_net.prob[k])
    return trans

"""
Coarsening Methods
"""

def multilevel_graph_coarsening(M, n_coarses, **kawgs):
    is_sequence = isinstance(n_coarses, np.ndarray) or isinstance(n_coarses, list) or isinstance(n_coarses, torch.Tensor)
    if is_sequence:
        Ms, Qs = [], []
        k = int(np.min(list(n_coarses)))
    else:
        k = int(n_coarses)

    n = M.num_nodes
    Gi = M.graph.detach().numpy()
    Q = np.eye(n)

    while n >= k: 
        stop_flag = 0
        max_dist = 10000
        max_dist_a = -1
        max_dist_b = -1
        for i in range(n):
            for j in range(i+1, n):
                dist = measure.normalized_L1(Gi[i], Gi[j])
                if dist < max_dist:
                    max_dist = dist
                    max_dist_a = i
                    max_dist_b = j

        if max_dist_a == -1 and n != 2: 
            max_dist_a, max_dist_b = random_two_nodes(n) 

        if n == 2 :
            max_dist_a, max_dist_b = 0, 1 

        cur_Q = merge_two_nodes(n, max_dist_a, max_dist_b) 
        Q = np.dot(Q, cur_Q) 

        Gi = multiply_Q(Gi, cur_Q) 

        idx_i = Q2idx(Q) 
        prob_i = np.array([list(idx_i).count(x) for x in set(idx_i)])
        prob_i = prob_i / np.sum(prob_i)

        n -= 1

        if is_sequence and n in n_coarses:
            Ms.append(M.transform(Q))
            Qs.append(lift_Q(Q))

    Mi = M.transform(Q)
    Q2 = torch.tensor(lift_Q(Q), dtype=torch.float32)

    if not is_sequence:
        return Mi, Q2
    else:
        print(len(Ms), len(n_coarses))
        assert(len(Ms) == len(n_coarses))
        return Ms, Qs

    
def weighted_graph_coarsening(M, n_coarses, seed=42, n_init=10, 
        sample_weight=None, init='k-means++', h_init=None, tol_empty=False):
    # S: the similarity mtx
    # n: the targeted number of clusters
    # scale: 0 return \barC S \barC.T; 1 return C S C.T; 2 return Cp S Cp.T
    is_sequence = isinstance(n_coarses, np.ndarray) or isinstance(n_coarses, list) or isinstance(n_coarses, torch.Tensor)
    if is_sequence:
        Ms, Qs = [], []
        k = int(np.min(list(n_coarses)))
    else:
        k = int(n_coarses)

    N, S = M.num_nodes, M.signless_laplacian()

    if k >= N: 
        Q = np.eye(N) 
        return M, Q/N # S, Q, Q2idx(Q) 

    def cluster(m):
        kmeans = KKMeans(n_clusters=m, n_init=n_init, init=init, 
                                        init_measure='inertia', random_state=seed) 
        idx = kmeans.predict(S, A=h_init, sample_weight=sample_weight, tol_empty=tol_empty) 
        if idx is None:
            print("specified initialization failed. turn to kmeans++") 
            kmeans = KKMeans(n_clusters=m, n_init=n_init, init="k-means++", 
                                        init_measure='inertia', random_state=seed) 
            idx = kmeans.predict(S, A=None, sample_weight=sample_weight, tol_empty=tol_empty) 
        
        Q = idx2Q(idx, m)

        Mc, Q2 = M.transform(Q, laplacian=True), torch.tensor(lift_Q(Q), dtype=torch.float32)
        return Mc, Q2

    if not is_sequence:
        res = cluster(k)

    else:
        Ms, Qs = list(zip(*[cluster(m) for m in n_coarses]))
        res = Ms, Qs 

    return res

def KGPC(m_net, n_coarse, cluster_mode=1, seed=1):
    num_nodes = m_net.num_nodes
    heuristic_sorted, common_neighbors_sorted_idx = sorted_heuristic(m_net, cluster_mode)
    
    dual_graph = torch.zeros((num_nodes, num_nodes))

    for w, (u,v) in zip(heuristic_sorted, common_neighbors_sorted_idx):
        dual_graph[u,v] = dual_graph[v,u] = w

    dual_graph = dual_graph.numpy()

    if cluster_mode == 0:
        clustering = SpectralClustering(n_clusters=n_coarse,
                assign_labels='discretize',
                affinity='precomputed',
                random_state=1).fit(dual_graph)

    else: 
        clustering = KMeans(n_clusters=n_coarse, random_state=seed).fit(dual_graph)


    labels = clustering.labels_
    
    mapping = {}

    for i, ci in enumerate(labels):
        if ci not in mapping:
            mapping[ci] = set([int(i)])
        else: 
            mapping[ci].add(int(i)) 

    mapping = [list(mapping[x]) for x in mapping if len(mapping[x]) > 0]
    n, n_coarse = m_net.num_nodes, len(mapping)

    Q = np.zeros((n, n_coarse))

    for i, x in enumerate(mapping):
        for y in x:
            Q[y, i] = 1 

    Mc, Q2 = m_net.transform(Q), torch.tensor(lift_Q(Q), dtype=torch.float32)
    
    return Mc, Q2

# def GPC(measure_net, n_coarse, seq=False, verbose=False, means_type=1): # GPC
def GPC(measure_net, n_coarses, verbose=False, means_type=1):
    is_sequence = isinstance(n_coarses, np.ndarray) or isinstance(n_coarses, list) or isinstance(n_coarses, torch.Tensor)
    if is_sequence:
        Ms, Qs = [], []
        k = int(np.min(list(n_coarses)))
    else:
        k = int(n_coarses)

    n = measure_net.num_nodes
    ni = measure_net.num_nodes
    Mi = measure_net
    
    Q = np.eye(n) # n x n, (n x k) (k x k - 1)

    while ni > k:
        best_idxs = sorted_heuristic(Mi, means_type)[1][0]
        vi, vf = int(torch.min(best_idxs)), int(torch.max(best_idxs))
        assert(vi != vf)
        if verbose:
            print(f"Merging nodes {vi} and {vf}")
        Q_curr = merge_two_nodes(ni, vi, vf)
        Q = np.dot(Q, Q_curr) 

        Mi = extract_from_mat(Mi, vi, vf)     # update measure net    
        ni -= 1 

        if is_sequence and (ni in n_coarses):
            Ms.append(Mi)
            Qs.append(Q)

    Q2 = true_trans((torch.tensor(Q) > 0).to(torch.float32), measure_net.prob.ravel())

    assert((Q.shape[0] == n) and (Q.shape[1] == k))
    assert(torch.abs(torch.sum(Q2)-1) < 1e-6)

    if is_sequence:
        res = Ms, Qs
    else:
        res = Mi, Q2

    return res

    # while num_nodes > n_coarse and (num_nodes > 1):
    #     heuristic_sorted, common_neighbors_sorted_idx = sorted_heuristic(curr_m_net, means_type)
    #     # print("heuristic list:\n", heuristic_sorted)
    #     for cs, vs in zip(heuristic_sorted, common_neighbors_sorted_idx):
    #         vi, vf = int(torch.min(vs)), int(torch.max(vs)) # make sure that node labels start from 0
    #         # if verbose and (vi != vf):
    #             # print(f"Merging {vi} and {vf}")
    #         if verbose and (vi == vf):
    #             print(f"Skipping self merge {vi} and {vf}")
    #             continue 

    #         cur_Q = merge_two_nodes(num_nodes, vi, vf)
    #         Q = np.dot(Q, cur_Q) 

    #         if trans:
    #             trans.append(get_trans(curr_m_net, vi, vf))

    #         next_m_net = extract_from_mat(curr_m_net, vi, vf)

    #         coarsenings.append(next_m_net)
    #         curr_m_net = next_m_net
    #         num_nodes -= 1
    #         break
    
    # if seq and trans:
    #     res = curr_m_net, Q, coarsenings, trans
    # elif seq:
    #     res = curr_m_net, coarsenings
    # else: 
    #     res = curr_m_net

    # Q2 = torch.tensor(lift_Q(Q))

    # return res, Q2

def spectral_graph_coarsening(mnet, n):
    G = mnet.graph.detach().numpy()
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n, n_init='auto').fit(v_all) 
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = idx2Q(idx, n)
            Gc = multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)

            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q

    Gc, Q2 = mnet.transform(Q_min), torch.tensor(lift_Q(Q_min), dtype=torch.float32)

    return Gc, Q2
    
class MeasureNetwork:
    def __init__(self, graph, prob=None, feat=None, grad=False): 
        assert(not (graph is None)) 

        self.graph = ensure_tensor(graph, grad)
        self.num_nodes = self.graph.shape[0] 

        if prob is None:
            self.prob = torch.ones((self.num_nodes, 1)) / self.num_nodes
        else:
            self.prob = ensure_tensor(prob)

        if feat is None:
            self.feat = torch.zeros((self.num_nodes, 1))
        else:
            self.feat = feat.reshape(self.num_nodes, -1)
            self.feat = ensure_tensor(self.feat)

        self.feat_dim = self.feat.shape[1]

    def transform(self, Q, laplacian=False): # Q is an assignment matrix: N x k
        A = self.signless_laplacian() if laplacian else self.graph.detach().numpy()
        X, Mu = self.feat.detach().numpy(), self.prob.detach().numpy()
        Q2 = lift_Q(Q)
        Ac = multiply_Q(A, Q2)
        Xc = Q2.T @ X
        Muc = Q.T @ Mu
        return MeasureNetwork(Ac, Muc, Xc)

    def get_graph(self, out_type=0):
        if out_type == 0:   # torch 
            return self.graph

        elif out_type == 1: # networkx
            return A_to_G(self.graph.numpy())

        else: # numpy
            return self.graph.numpy()
    
    def get_prob(self):
        return self.prob
    
    def get_feat(self):
        return self.feat

    def get_all(self):
        return self.graph, self.prob, self.feat

    def comp_dist(self, other, alpha=0, ot_layers=30, sinkhorn_layers=5, 
                  noise_lvl=7, eps=1e-8, tries=20, verbose=False, sequence=False):
        
        res = fgwd(self, other, alpha=alpha, ot_layers=ot_layers, 
             sinkhorn_layers=sinkhorn_layers, noise_lvl=noise_lvl, eps=eps, 
             tries=tries, verbose=verbose, sequence=sequence)
        return res

    def distortion(self, other, tran, cost_mode=0, alpha=0):
        G1, P1, F1 = self.graph, self.prob, self.feat
        G2, P2, F2 = other.graph, other.prob, other.feat
        C = cost_mat(G1, G2, P1, P2, F1, F2, tran, cost_mode, alpha)
        dist = (C * tran).sum()
        return dist

    def plot(self, ax=None, nodesize=1000, color='k', labels=False):
        graph_nx = A_to_G(self.graph.data.numpy())
        widths = [graph_nx[u][v]['weight'] for (u, v) in graph_nx.edges()]
        node_sizes = [nodesize*self.prob[i] for i in range(self.num_nodes)]
        pos = nx.spring_layout(graph_nx)
        nx.draw_networkx(graph_nx, width=widths, node_size=node_sizes, ax=ax, 
                         with_labels=labels, node_color=color, pos=pos)

    def signless_laplacian(self):
        G = self.graph.data.numpy()
        D_12 = np.diag(np.sum(G, axis=1)**(-1/2))
        S = np.eye(G.shape[0]) + D_12 @ G @ D_12
        return S

    def coarsen(self, n_coarse, coarse_mode, means_type=1, seed=1, verbose=False):
        is_seq = True in [isinstance(n_coarse, x) for x in [list, np.ndarray, torch.Tensor]]
        if is_seq: # sequence of coarsenings
            n_coarse = [int(x) for x in n_coarse]
        else:
            n_coarse = int(n_coarse)

        if coarse_mode == 0: # Jin Multilevel Method 
            res = multilevel_graph_coarsening(self, n_coarse) 
        elif coarse_mode == 1: # Our Iterative coarsening 
            res = GPC(self, n_coarse, verbose, means_type) 
        elif coarse_mode == 2: # Chen method with signless laplacian 
            res = weighted_graph_coarsening(self, n_coarse, seed) 
        elif coarse_mode == 3: # Our Spectral Coarsening method 
            if is_seq: 
                res = [KGPC(self, n, means_type, seed) for n in n_coarse] 
                res = list(zip(*res)) # Ms and Qs 
            else: 
                res = KGPC(self, n_coarse, means_type, seed) 
        elif coarse_mode == 4: # Jin Spectral Coarsening 
            if is_seq: 
                res = [spectral_graph_coarsening(self, n) for n in n_coarse] 
                res = list(zip(*res)) # Ms and Qs 
            else:
                res = spectral_graph_coarsening(self, n_coarse)

        else:
            raise(ValueError)

        return res

    # def coarsen(self, n_coarse, coarse_mode, levels=None, trans=False, means_type=1, verbose=False, seed=1):
    #     if not levels is None:
    #         coarse_lvls = [int(np.ceil(alpha * self.num_nodes)) for alpha in levels]
    #         coarse_lvls.sort(reverse=True)
    #         n_coarse = np.min(coarse_lvls)
    #     else:
    #         coarse_lvls = None

        # if coarse_mode == 0: # Jin Multilevel Method 
        #     res = multilevel_graph_coarsening(self, n_coarse, coarse_lvls)
        # elif coarse_mode == 1: # Our Iterative coarsening
        #     res = gw_coarsening(self, n_coarse, coarse_lvls, trans, verbose, means_type)
        # elif coarse_mode == 2: # Chen method with signless laplacian
        #     if not coarse_lvls is None:
        #         res = [weighted_graph_coarsening(self, n) for n in coarse_lvls]
        #         res = list(zip(*res)) # Ms and Qs
        #     else:
        #         res = weighted_graph_coarsening(self, n_coarse)
        # elif coarse_mode == 3: # Our Spectral Coarsening method
        #     if not coarse_lvls is None:
        #         res = [gw_spec_coarsening(self, n, means_type, seed) for n in coarse_lvls]
        #         res = list(zip(*res)) # Ms and Qs
        #     else:
        #         res = gw_spec_coarsening(self, n_coarse, means_type, seed)
        # elif coarse_mode == 4: # Jin Spectral Coarsening
        #     if not coarse_lvls is None:
        #         res = [spectral_graph_coarsening(self, n) for n in coarse_lvls]
        #         res = list(zip(*res)) # Ms and Qs
        #     else:
        #         res = spectral_graph_coarsening(self, n_coarse)

        # return res


