import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '../../core')
from DataTools import MeasureNetwork

import random
import numpy as np
import pandas as pd
import networkx as nx
import ot
from clf_helper import *
from tqdm import tqdm
from DataTools import weighted_graph_coarsening
import matplotlib.pyplot as plt
import torch
from DataTools import sorted_heuristic

import os
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

from methods.DataIO import StructuralDataSampler
import methods.FusedGromovWassersteinFactorization as FGWF
import dev.util as util
from CoarsenTools import make_dataset
from sklearn.metrics import rand_score

def train(dataset_name, method_name, base='./', folder='data_coarsen_40'):
    coarsened_data, num_classes = make_dataset(dataset_name, method_name, base, folder)
    if num_classes > 2:
        print("MORE THAN 2 CLASSES!")

    sizes = [x[1] for x in coarsened_data] 
    avg_size, std_size = np.mean(sizes), np.std(sizes)

    # model params
    num_atoms = 15
    size_atoms = np.ceil(avg_size + np.random.randn(num_atoms) * std_size).astype(int) 
    size_atoms = np.minimum(15, size_atoms) 
    ot_method = 'ppa'
    # ot_method = 'b-admm'
    gamma = 1e-1
    gwb_layers = 5
    ot_layers = 50

    # alg. params
    size_batch = 25
    epochs = 10
    lr = 0.05
    weight_decay = 0
    shuffle_data = True
    zeta = None  # the weight of diversity regularizer
    mode = 'fit'
    best_acc = float('inf')

    dim_embedding = 1
    data_sampler = StructuralDataSampler(coarsened_data)
    labels = []

    for sample in coarsened_data:
        labels.append(sample[-1])
    labels = np.asarray(labels)
    if -1 in labels:
        labels = (labels + 1)/2
    if 2 in labels:
        labels = labels - 1

    model = FGWF.FGWF(num_samples=len(coarsened_data),
                    num_classes=num_classes,
                    size_atoms=size_atoms,
                    dim_embedding=dim_embedding,
                    ot_method=ot_method,
                    gamma=gamma,
                    gwb_layers=gwb_layers,
                    ot_layers=ot_layers,
                    prior=data_sampler)
    

    model = FGWF.train_usl(model, 
                           coarsened_data,
                           size_batch=size_batch,
                           epochs=epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           shuffle_data=shuffle_data,
                           zeta=zeta,
                           mode=mode,
                           visualize_prefix=os.path.join(util.RESULT_DIR, dataset_name + '_' +method_name))
    model.eval()
    features = model.weights.cpu().data.numpy()
    embeddings = features.T
    kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
    pred = kmeans.fit_predict(embeddings)

    score = rand_score(labels, pred)

    if num_classes == 2:

        best_acc = max([1 - np.sum(np.abs(pred - labels)) / len(coarsened_data),
                    1 - np.sum(np.abs((1 - pred) - labels)) / len(coarsened_data)])

        print(f"Best Acc  : {best_acc}")
        print(f"Rand Score: {score}")

    else: 
        score = rand_score(labels, pred)
        print(f"Rand Score: {score}")

    return score


if __name__ == "__main__":
    # dataset_opts = ["ENZYMES", "IMDB-BINARY", "MSRC_9", "MUTAG", "NCI1", "NCI109", "PROTEINS", "PTC_MR", "tumblr_ct1"] 
    datasets = ["MUTAG", "IMDB-BINARY", "ENZYMES", "PTC_MR", "MSRC_9"] # "[PROTEINS]"
    methods =  ["Jin_Multi", "Jin_Spectral", "Chen_GW", "Our_Spectral", "Our_Iter", "Original"] # Do ENZYMES

    n_dataset, n_method = int(sys.argv[1]), int(sys.argv[2])
    dataset_name, method_name = datasets[n_dataset], methods[n_method]

    dir1 = "./cluster_results_final"
    dir2 = os.path.join(dir1, dataset_name)
    dir3 = os.path.join(dir2, method_name)

    for path in [dir1, dir2, dir3]:
        if not os.path.isdir(path):
            os.mkdir(path)

    num_logs = len(os.listdir(dir3))
    log_file_path = os.path.join(dir3, f"log{num_logs}.log") 

    log_file = open(log_file_path,"w") 
    sys.stdout = log_file
    
    base = './'
    folder = 'dataset_coarsen_40'

    score = train(dataset_name, method_name, base, folder)

    print("Done!")
