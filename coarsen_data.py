import sys
sys.path.insert(0, './core')
# from DataTools import MeasureNetwork
from tqdm import tqdm
from CoarsenTools import coarsen_data, parse_dataset
import os
import torch
import numpy as np

if __name__ == "__main__":
    # datasets = ["ENZYMES", "IMDB-BINARY", "PROTEINS", "PTC_MR"] # "MUTAG", # "MSRC_9"
    # datasets = ["AIDS", "IMDB-M", "PTC-MR", "PROTEINS", "ENZYMES"]
    # datasets = ["tumblr_ct1", "PROTEINS"]#, "PTC_MR"] 
    # datasets = ["tumblr_ct1", "PROTEINS", "ENZYMES", "PTC_MR", "MSRC_9"] #  "MUTAG", "IMDB-BINARY",
    # datasets = ["PTC_MR"]#, "PROTEINS", 'ENZYMES']
    datasets = ["PTC_MR", "PROTEINS", "ENZYMES"]
    methods =  ["Jin_Multi", "Jin_Spectral", "Chen_GW", "Our_Spectral", "Our_Iter"] # Do ENZYMES
    dir_prefix = 'coarsened_data_spectral'
    # coarse_levels = list(np.linspace(0, 1, 21)[-2:0:-1]) 
    coarse_levels = 0.4
    cpus = 1
    for dataset_name in datasets: # have to do proteins data with our iter algorithm
        for method in tqdm(range(5)): #tqdm(range(5)): # [0, 1, 2, 3, 4]: # Run this for iterative algorithm too 
            method_name = methods[method] # figure out what's going on with 
            base = os.path.join('.', 'dataset') 
            data = parse_dataset(base, dataset_name) 
            coarsen_data(data, dataset_name, method, method_name, coarse_levels, cpus, './dataset_exp1_2') 
