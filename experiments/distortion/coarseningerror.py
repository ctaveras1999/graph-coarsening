import numpy as np
import sys
sys.path.insert(1, '../../core/')
sys.path.insert(1, '../../')
sys.path.insert(1, '.')
sys.path.insert(1, './core')
import networkx as nx
import os
import scipy.sparse as sps
from DataTools import MeasureNetwork
from tqdm import tqdm
from CoarsenTools import parse_dataset, load_data
import torch 

methods = ["MGC", "WGC", 'KGPC', 'GPC'] #"SGC"]#"WGC", "KGPC", "GPC"]
datasets = ['IMDB-BINARY'] #['ENZYMES']# ['MSRC_9', 'MUTAG', 'PTC_MR']

# FIX WGC, GPC!! 

if __name__ == "__main__":
    base = '../..'
    folder = 'coarsened_final'
    
    for i, dataset_name in enumerate(datasets):
        print(dataset_name)
        A, P, X, _ = load_data(dataset_name, "Original", base, False)
        is_seq = isinstance(A[0], list) or isinstance(A[0], np.ndarray)
        seq_len = len(A) 
        Ac,_,_,_,_ = load_data(dataset_name, methods[0], base, folder, True)
        seq_len = len(Ac[0])

        Ms = [MeasureNetwork(*x) for x in zip(A,P,X)]
        num_dists_comp = (len(methods), len(A), seq_len) if is_seq else (len(methods), len(len(A)))
        all_dists = torch.zeros(num_dists_comp)

        # print("Shape:", all_dists.shape)

        for j, method_name in enumerate(methods): 
            Ac,Pc,Xc,Q,labels = load_data(dataset_name, method_name, base, folder, True)
            # print(Ac[0])
            # for ac, xc in zip(Ac[18], Xc[18]):
                # print(method_name, ac.shape, xc.shape, len(Ac[18]), len(Xc[18]))
            # print(Ac[18][0])
            # print(Ac[18][1])
            if not is_seq: 
                Mcs = [MeasureNetwork(*x) for x in zip(Ac,Pc,Xc)]
                for k, (m, mc, q) in enumerate(zip(Ms, Mcs, Q)): 
                    tran = torch.tensor(q>0, dtype=m.graph.dtype) / q.shape[0]
                    if j == 2: 
                        mc = m.transform(torch.tensor(q>0, dtype=m.graph.dtype)) # to make fairer comparison, compare adjacencies
                    assert(torch.abs(1 - torch.sum(tran)) < 1e-4)
                    dist = m.distortion(mc, tran)  
                    all_dists[j, k] = dist 
            else: 
                for k in range(len(Ac)):
                    # print(j,k,Ac[k][0].shape, Pc[k][0].shape, Xc[k][0].shape)
                    Ms_Qs = [[MeasureNetwork(*m) , q] for (m,q) in zip(zip(Ac[k], Pc[k], Xc[k]), Q[k])]
                    m = Ms[k]
                    for l, (mc, q) in enumerate(Ms_Qs): 
                        q_prime = torch.tensor(q>0, dtype=m.graph.dtype)
                        tran = q_prime / q.shape[0]
                        if j == 2: 
                            mc = m.transform(q_prime) # to make fairer comparison, compare adjacencies
                        assert(torch.abs(1 - torch.sum(tran)) < 1e-4)
                        dist = m.distortion(mc, tran)
                        # print(j,k,l)
                        all_dists[j,k,l] = dist

        if not os.path.isdir('results'):
            os.mkdir('results')
        np.save(f'./results/dists_{dataset_name}', all_dists.numpy()) 