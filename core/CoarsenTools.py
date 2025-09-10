import multiprocessing
import numpy as np 
import os 
from DataTools import MeasureNetwork
import networkx as nx
import scipy.sparse as sps
import torch


def parse_dataset(dir, DS):
    prefix = dir + '/' + DS + '/' + DS
    A = prefix + '_A.txt' 
    offsets = np.loadtxt(prefix +'_graph_indicator.txt', dtype=int, delimiter=',') - 1
    offs = np.append([0], np.append(np.where((offsets[1:] - offsets[:-1])>0)[0]+1, len(offsets)))
    labels = np.loadtxt(prefix+'_graph_labels.txt', dtype=np.float64).reshape(-1)
    A_data = np.loadtxt(prefix+'_A.txt', dtype=int, delimiter=',') - 1
    A_mat = sps.csr_matrix((np.ones(A_data.shape[0]), (A_data[:, 0], A_data[:, 1])), dtype=int)
    
    As = []
    for i in range(1, len(offs)):
        As.append(A_mat[offs[i-1]:offs[i],offs[i-1]:offs[i]])

    am = [np.array(sps.csr_matrix.todense(x.astype(np.float64))) for x in As]
    am_corrected = []
    label_corrected = []
    N = len(am)
    for i in range(N):
        d = sum(am[i], 0)
        if not np.any(d == 0):
            am_corrected.append((am[i]>0).astype(float))
            label_corrected.append(labels[i])
            # node_label_corrected.append(node_labels[i])

    Xs = [None] * len(am_corrected)
    node_label_corrected = None

    return am_corrected, Xs, label_corrected, node_label_corrected

def load_data(dataset_name, method_name, base='../../', folder='dataset_coarse', trans=False): 
    if method_name != "Original":
        base = os.path.join(base, folder)
        base = os.path.join(base, dataset_name, method_name) 
        print("Current Working Directory:")
        print(os.getcwd())
        print()
        n = len([x for x in os.listdir(base) if "A_" in x]) 
        A_file = os.path.join(base, f'A_{n}.npy') 
        P_file = os.path.join(base, f'P_{n}.npy') 
        X_file = os.path.join(base, f'X_{n}.npy') 
        label_file = os.path.join(base, 'graph_labels.npy') 
        A, P, X, labels = [np.load(x, allow_pickle=True) for x in [A_file, P_file, X_file, label_file]] 
    else: 
        base = os.path.join(base, 'dataset')
        A, X, labels, _ = parse_dataset(base, dataset_name) 
        P = np.array([np.ones(a.shape[0])/a.shape[0] for a in A], dtype=object)
    
    if not trans:
        return A, P, X, labels
    
    else:
        Q_file = os.path.join(base, f'Q_{n}.npy')
        Q = np.load(Q_file, allow_pickle=True)
        return A, P, X, Q, labels

def make_dataset(dataset_name, method_name, base='../..', folder='dataset_coarse', trans=False):
    A,P,X,labels = load_data(dataset_name, method_name, base, folder, trans)
    num_classes = len(set(labels))
    coarsened_data = []

    for a,p,x,l in zip(A, P, X, labels):
        Nc = a.shape[0]
        edges_c = np.array(np.where(a > 0)).T
        weights = np.array([a[*y] for y in edges_c]).reshape(-1, 1)
        edges_c = np.hstack([edges_c, weights])
        coarsened_data.append([edges_c, Nc, l])

    return coarsened_data, num_classes

def process(As, Xs=None):
    A_post, X_post = [None] * len(As), [None] * len(As)
    for i in range(len(As)):
        a, x = As[i], None if Xs is None else Xs[i]
        Gi = nx.from_numpy_array(a)
        idxs = list(max(nx.connected_components(Gi), key=len))
        Gc = a[idxs]
        Gc = Gc[:,idxs]
        Xc = x[idxs] if x is not None else None
        A_post[i], X_post[i] = Gc, Xc
    
    P_post = [None] * len(A_post)

    return A_post, P_post, X_post

def get_opts(method, redu_factor):
    if method == 0: # Jin (MGC) 
        opts = [redu_factor, 0, 1, 1, False] # coarse_mode, normalization_mode=2, seq=False, trans=False, means_type=1, verbose=False, seed=1
    elif method == 1: # Zeng (KGC)
        opts = [redu_factor, 4, 1, 1, False] # n_coarse, coarse_mode, normalization_mode=2, levels=None, trans=False, means_type=1, verbose=False, seed=1)
    elif method == 2: # Jin (SGC)
        opts = [redu_factor, 2, 1, 1, False] #  coarse_mode, levels=None, trans=False, means_type=1, verbose=False, seed=1
    elif method == 3: # Spectral
        opts = [redu_factor, 3, 1, 1, False] #  n_coarse, coarse_mode, means_type=1, seed=1, verbose=False)
    elif method == 4: # Iterative 
        opts = [redu_factor, 1, 1, 1, False] 
    else: 
        raise(ValueError) 
    return opts

def node_fn(levels, n_min, n_max): 
    f = lambda x: int(np.clip(np.ceil(n_max * x), n_min, n_max))

    if isinstance(levels, np.ndarray) or isinstance(levels, list) or isinstance(levels, torch.Tensor):
        n_levels = [f(c) for c in levels]
    else: 
        n_levels = f(levels)
    return n_levels

coarsen = lambda G, opts: G.coarsen(node_fn(opts[0], 2, G.num_nodes), *opts[1:])

def coarsen_mat(M, opts):
    res, Q = coarsen(MeasureNetwork(*M), opts)
    if isinstance(res, list) or isinstance(res, tuple):
        As, Ps, Xs, = [], [], []
        for m in res: 
            As.append(m.graph.detach().numpy())
            Ps.append(m.prob.detach().numpy())
            Xs.append(m.feat.detach().numpy())
            res = As, Ps, Xs, Q
    else: 
        # print(f"Type = {type(res)}")
        assert(isinstance(res, MeasureNetwork))
        res = res.graph.detach().numpy(), res.prob.detach().numpy(), res.feat.detach().numpy(), Q
    return res


def coarsen_data(data, dataset_name, method, method_name, redu_factor, cpus, base_dir='./dataset_coarse'):
    # methods = 1) Jin et al, 2) Chen et al, 3) ours iterative, 4) ours k-means
    As, Xs, graph_labels, _ = data # get connected component?
    
    num_graphs = len(As)
    Ps = [None] * num_graphs 
    opts = get_opts(method, redu_factor)
    opts_list = [opts] * len(As) 

    with multiprocessing.Pool(cpus) as pool:
        res = pool.starmap(coarsen_mat, zip(zip(As, Ps, Xs), opts_list)) # change back! 

    Acs, Pcs, Xcs, Qs = np.zeros(num_graphs, dtype=object), np.zeros(num_graphs, dtype=object), np.zeros(num_graphs, dtype=object), np.zeros(num_graphs, dtype=object)

    for i, res_i in enumerate(zip(res)):
        if len(res_i[0]) < 4:
            print("ERROR:")
            print(res_i)
            print(res_i[0])
            continue
        A, P, X, Q = res_i[0]
        Acs[i] = A
        Pcs[i] = P
        Xcs[i] = X
        Qs[i] = Q

    res_dir1 = f"{base_dir}/{dataset_name}"
    res_dir2 = f"{res_dir1}/{method_name}"

    for res_dir in [base_dir, res_dir1, res_dir2]:
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)

    np.save(f"{res_dir2}/graph_labels.npy", graph_labels)

    if not os.path.isdir(res_dir2):
        os.mkdir(res_dir2)
        np.save(f"{res_dir2}/graph_labels.npy", graph_labels)
    
    ver = len([x for x in os.listdir(res_dir2) if 'A_' in x]) + 1

    np.save(f"{res_dir2}/A_{ver}.npy", Acs)
    np.save(f"{res_dir2}/X_{ver}.npy", Xcs)
    np.save(f"{res_dir2}/P_{ver}.npy", Pcs)
    np.save(f"{res_dir2}/Q_{ver}.npy", Qs)
    return res, graph_labels
