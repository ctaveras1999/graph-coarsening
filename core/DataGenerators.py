import numpy as np
import networkx as nx
import FGWF
import torch
from GraphUtils import A_to_G
from DataTools import MeasureNetwork

def add_edge_weights(Gs):
    for G in Gs:
        for (u,v) in G.edges():
            G[u][v]['weight'] = G[v][u]['weight'] = 1
    return Gs

def nx_graph(i, N, k=2):
    G = None
    if i == 0:
        G = nx.star_graph(N-1)
    elif i == 1:
        G = nx.cycle_graph(N)
    elif i == 2:
        G = nx.binomial_tree(int(np.log2(N)))
    elif i == 3:
        G = nx.full_rary_tree(k, N)
    elif i == 4:
        G = nx.turan_graph(N, k)
    elif i == 5:
        G = nx.complete_multipartite_graph(N//2-1,N//2+1)
    elif i == 6:
        G = nx.circular_ladder_graph(N//2)
    elif i == 7:
        G = nx.ladder_graph(N//2)
    elif i == 8:
        G = nx.wheel_graph(N)

    G = add_edge_weights([G])[0]
    return G

def gw_interp(N, G_idxs, block_size, mode, gwb_layers=20, ot_layers=20, starting_idx=0):
    cost_mode, embed_dim = 0, 1
    num_atoms = len(G_idxs)
    G_basis = [nx_graph(i, N) for i in G_idxs]

    weights = np.zeros((num_atoms, (block_size-1)* num_atoms+1))
    for i in range(num_atoms):
        start = (block_size-1) * i
        end = start + block_size
        weights[i,start:end] = np.linspace(1,0,block_size)
        weights[(i+1)%num_atoms, start:end] = np.linspace(0,1,block_size)
    
    eps = 1e-16
    atom_priors = [torch.Tensor(nx.to_numpy_array(x)) for x in G_basis]

    model = FGWF.FGWF(num_samples= num_atoms * (block_size - 1),
                    num_classes=num_atoms,
                    size_atoms=[N] * num_atoms,
                    dim_embedding=embed_dim,
                    ot_method='ppa',
                    gwb_layers=gwb_layers,
                    ot_layers=ot_layers,
                    atom_prior=atom_priors,
                    weight_prior=torch.Tensor(weights).T,
                    alpha=0,
                    cost_mode=cost_mode)

    G_seq = []
    for i, wi in enumerate(weights.T):
        Gi_init = None
        if mode == 0:
            Gi_init = model.output_atoms(np.argmax(wi)) if (i==0) else G_seq[-1]
        elif mode == 1:
            Gi_init = model.output_atoms(np.argmax(wi))
        elif mode == 2:
            Gi_init = model.output_atoms(starting_idx)
        Gi = FGWF.gen_graph(model, Gi_init, wi, False, 1)
        G_seq.append(Gi)
    
    G_seq = [A_to_G(x.detach().numpy()) for x in G_seq]

    return G_seq, weights, G_basis, model

def erdos_renyi(N, ps):
    return add_edge_weights([nx.erdos_renyi_graph(N, p) for p in ps])

def sbm(N, blocks, ps_intra, ps_inter):
    mats = []
    blocks_list = None
    if isinstance(blocks, float) or isinstance(blocks, int):
        blocks_list = [blocks] * len(ps_inter)
    else:
        blocks_list = blocks
    if isinstance(ps_intra, float) or isinstance(ps_intra, int):
        ps_intra = [ps_intra] * len(ps_inter)
    else:
        ps_intra = ps_intra

    for (p_in, p_out, block) in zip(ps_intra, ps_inter, blocks_list):
        prob_mat = np.ones((block, block)) * p_out
        for i in range(block):
            prob_mat[i,i] = p_in
        mat = nx.stochastic_block_model([N] * block, prob_mat)
        mats.append(mat)
    return add_edge_weights(mats)

def construct_community(A, active_edges, p):
    active_edges = {} if active_edges is None else active_edges
    N = A.shape[0]
    exp_edges = np.round(N*(N-1)/2*p)
    num_edges = np.sum(A)//2
    node_set = np.arange(N)
    while num_edges < exp_edges:
        v1, v2 = np.sort(np.random.choice(node_set, 2, False))
        if v1 not in active_edges:
            active_edges[v1] = set([v2])
            A[v1][v2] = A[v2][v1] = 1
            num_edges += 1
        elif v1 in active_edges and v2 not in active_edges[v1]:
            active_edges[v1].add(v2)
            A[v1][v2] = A[v2][v1] = 1
            num_edges += 1
        else:
            continue
    return A, active_edges
        
def sbm_ts(N, M, pin, pout):
    As = [np.zeros((N*M, N*M)) for _ in pout]
    active_edges = [None] * len(pout)
    for i in range(M):
        for j in range(M):
            t,l,b,r = i*N, j*N, (i+1)*N, (j+1)*N
            Aij = np.zeros((N,N))
            active_edges = None
            if i == j:
                A, _ = construct_community(Aij, None, pin)
                for k in range(len(pout)):
                    As[k][t:b,l:r] = A
            else:
                for k, pk in enumerate(pout):
                    Aij, active_edges = construct_community(Aij, active_edges, pk)
                    As[k][t:b,l:r] = Aij
    return [A_to_G(x) for x in As]

def watts_strogatz(N, ms, ps):
    if isinstance(ms, float) or isinstance(ms, int):
        ms_list = [ms] * len(ps)
    else:
        ms_list = ms
    zipped = zip(ms_list, ps)
    graphs = [nx.watts_strogatz_graph(N, m, p) for (m,p) in zipped]
    return add_edge_weights(graphs)

def make_data(data_type: int, 
              N: int = 8, 
              ps = None, 
              pin = None,
              ms = 3, 
              blocks: int = 3, 
              block_size: int = 20, 
              interp_mode: int = 0, 
              gamma: float = 5e-2, 
              gwb_layers: int = 5, 
              ot_layers: int = 10,
              starting_idx: int = 0,
              idxs = None):
    data_fields = {}

    if (data_type != 3) and (ps is None):
        ps = np.linspace(0, 1, block_size)

    if data_type == 0: # sbm
        pin = 1/2 if pin is None else pin
        Gs = sbm(N, blocks, pin, ps)
        data_fields["name"] = "sbm"
        data_fields["N"] = N * blocks
        data_fields["pout"] = ps
        data_fields["pin"] = pin
        data_fields["Gs"] = Gs
        data_fields["blocks"] = blocks

    if data_type == 1: # sbm-ts
        pin = 1/2 if pin is None else pin
        Gs = sbm_ts(N, blocks, pin, ps)
        data_fields["name"] = "sbm-ts"
        data_fields["N"] = N * blocks
        data_fields["pout"] = ps
        data_fields["pin"] = pin
        data_fields["Gs"] = Gs
        data_fields["blocks"] = blocks

    elif data_type == 2: # Erdos-Renyi
        Gs = erdos_renyi(N, ps)
        data_fields["name"] = "erdos-renyi"
        data_fields["N"] = N
        data_fields["Gs"] = Gs
        data_fields["ps"] = ps

    elif data_type == 3: # Watts Strogatz
        Gs = watts_strogatz(N, ms, ps)
        data_fields["name"] = "watts-strogatz"
        data_fields["ms"] = ms
        data_fields["ps"] = ps
        data_fields["Gs"] = Gs

    elif data_type == 4: # Artificial
        if idxs is None:
            idxs = [0, 1, 2]
        Gs, Ws, G_basis, model = gw_interp(N, idxs, block_size, interp_mode, gamma, gwb_layers, ot_layers, starting_idx)
        data_fields["name"] = "artificial"
        data_fields["N"] = N
        data_fields["block_size"] = block_size
        data_fields["weights"] = Ws
        data_fields["Gs"] = Gs
        data_fields["basis"] = G_basis
        data_fields["model"] = model

    return data_fields, Gs

def make_sbm(num_nodes, num_blocks, pin, pout, seed=None):
    left_overs = num_nodes % num_blocks
    block_sizes = [num_nodes // num_blocks + (1 if i < left_overs else 0) for i in range(num_blocks)]
    P = np.eye(num_blocks) * pin + pout * (np.ones((num_blocks, num_blocks)) - np.eye(num_blocks))
    G = nx.stochastic_block_model(block_sizes, P, seed=seed)
    for (vi, vf) in G.edges():
        G[vi][vf]['weight'] = 1
        G[vf][vi]['weight'] = 1

    return G

def get_sbm_data(num_graphs, node_ranges, num_blocks, pin, pout, seed=None):
    sampleRNG = np.random.default_rng(seed)
    if isinstance(num_blocks, float):
        num_blocks = [num_blocks] * num_graphs
    num_nodes  = sampleRNG.choice(np.arange(*node_ranges), num_graphs, True)
    Gs = [make_sbm(x, y, pin, pout, seed+i) for i, (x,y) in enumerate(zip(num_nodes, num_blocks))]
    m_nets = [MeasureNetwork(x, None, None) for x in Gs]
    return m_nets


def get_boundaries(nodes_per_block):
    N = len(nodes_per_block)
    first, last = np.zeros((1,1)), np.ones((1,1)) * (N+1)
    mid = np.cumsum(nodes_per_block).reshape(-1, 1)
    bds = np.vstack([first, mid, last]).reshape(-1).astype(int)
    return bds

def make_block_graph(nodes_per_block, pin, pout, stochastic=False, mixed=False):
    num_blocks = len(nodes_per_block)
    N = np.sum(nodes_per_block)
    bds = get_boundaries(nodes_per_block)
    X = np.zeros((N, N))

    for i in range(num_blocks):
        for j in range(i, num_blocks):
            t, b = bds[i], bds[i+1]
            l, r = bds[j], bds[j+1]
            if i != j:
                block_shape = [(b-t), (r-l)] 
                if (not stochastic) and (not mixed):
                    X_ij = np.zeros(block_shape[0] * block_shape[1]) 
                    num_ones = np.ceil((b-t) * (r-l) * pout).astype(int)
                    X_ij[:num_ones] = 1
                    np.random.shuffle(X_ij)

                elif stochastic: # true sbm
                    X_ij = (np.random.uniform(0, 1, size=block_shape) <= pout).astype(float)
                
                elif mixed:
                    num_elems = (b-t) * (r-l)
                    P_ij = np.random.choice([pin, pout], 1)
                    Y_ij = np.random.uniform(0, 1, size=num_elems)
                    X_ij = (Y_ij <= P_ij).astype(float)

                X[t:b,l:r] = X_ij.reshape(b-t, r-l)
                X[l:r,t:b] = X_ij.reshape(b-t, r-l).T

            else:
                n = b-t
                n_choose_2 = int(n * (n-1) / 2)
                if (not stochastic) and (not mixed):
                    X_ij = np.zeros((n_choose_2))
                    num_ones = np.ceil(n_choose_2 * pin).astype(int)
                    X_ij[:num_ones] = 1
                    np.random.shuffle(X_ij)
                elif stochastic: 
                    X_ij = (np.random.uniform(0, 1, size=n_choose_2) <= pin).astype(float)
                elif mixed:
                    num_elems = int((b-t) * (b-t-1) / 2)
                    P_ij = np.random.choice([pin, pout], 1)
                    Y_ij = np.random.uniform(0, 1, size=num_elems)
                    X_ij = (Y_ij <= P_ij).astype(float)

                triu_idxs = np.triu_indices(n, k = 1)
                triu_idxs = triu_idxs[0] + t, triu_idxs[1] + l
                X[triu_idxs] = X_ij.reshape(-1)
                X[triu_idxs[::-1]] = X_ij.reshape(-1)

    return MeasureNetwork(X)

def random_block_graph(pin=0.5, pout=0.5, block_range=[1,6], node_range=[1,6], stochastic=False):
    num_blocks = np.random.randint(*block_range)
    nodes_per_block = np.random.randint(*node_range, size=num_blocks)
    return make_block_graph(nodes_per_block, pin, pout, stochastic), nodes_per_block

def random_block_graphs(num_graphs, pin=1, pout=0.05, block_range=[1,6], node_range=[1,6], stochastic=False):
    return [random_block_graph(pin, pout, block_range, node_range, stochastic) for _ in range(num_graphs)]

import random
import numpy as np
import numpy.linalg as LA
from scipy import sparse

def erdos(n, p):
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if tmp <= p:
                G[i, j] = 1
                G[j, i] = 1
    sum_G = np.sum(G, 1)
    for i in range(n):
        if sum_G[i] == 0:
            j = random.randint(0, n-1)
            G[i, j] = 1
            G[j, i] = 1
    return G


def sbm_pq(n, k, p, q):
    G = np.zeros((n, n))
    block_size = n//k
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if i//block_size == j // block_size:
                if tmp <= p:
                    G[i, j] = 1
                    G[j, i] = 1
            else:
                if tmp <= q:
                    G[i, j] = 1
                    G[j, i] = 1
    # sum_G = np.sum(G, 1)
    for i in range(n):
        G[i,i] = 0
        if np.sum(G[i]) == 0:
            j = random.randint(0, n-1)
            G[i, j] = 1
            G[j, i] = 1
    return G

def sbm_qp(n, k, p, q):
    G = sbm_pq(n, k, q, p)
    return G

def sbm_pq_mixed(n, k, p, q):
    G = np.zeros((n, n))
    block_size = n//k
    B = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            tmp = random.random()
            if tmp < 1/2:
                B[i, j] = p
                B[j, i] = p
            else:
                B[i, j] = q
                B[j, i] = q
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if tmp <= B[i//block_size, j // block_size]:
                G[i, j] = 1
                G[j, i] = 1

    # sum_G = np.sum(G, 1)
    for i in range(n):
        G[i,i] = 0
        if np.sum(G[i]) == 0:
            j = random.randint(0, n)
            j = random.choice([k for k in range(n) if k != i])
            G[i, j] = 1
            G[j, i] = 1

    return G
