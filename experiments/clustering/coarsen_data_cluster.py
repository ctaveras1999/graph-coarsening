import sys
sys.path.insert(0, '../../core')
sys.path.insert(0, '../../')

from tqdm import tqdm
from CoarsenTools import coarsen_data, parse_dataset
import os

if __name__ == "__main__":
    # datasets = ["ENZYMES", "IMDB-BINARY", "PROTEINS", "PTC_MR"] # "MUTAG", # "MSRC_9"
    # datasets = ["tumblr_ct1", "PROTEINS"]#, "PTC_MR"] 
    # datasets = ["tumblr_ct1", "PROTEINS", "ENZYMES", "PTC_MR", "MSRC_9"] #  "MUTAG", "IMDB-BINARY",
    # datasets = ["PTC_MR"]#, "PROTEINS", 'ENZYMES']
    # datasets = ["PTC_MR", "PROTEINS", "ENZYMES"]
    # datasets = ["IMDB-BINARY", "MUTAG", "PROTEINS", "MSRC_9", "ENZYMES", "PTC_MR"]
    methods =  ["Jin_Multi", "Jin_Spectral", "Chen_GW", "Our_Spectral", "Our_Iter"] # Do ENZYMES
    datasets = ["PROTEINS"]
    dir_prefix = 'coarsened_data_spectral'
    coarse_levels = 0.4
    cpus = 50
    for dataset_name in datasets: # have to do proteins data with our iter algorithm
        for method in tqdm([4]): #tqdm(range(5)): # [0, 1, 2, 3, 4]: # Run this for iterative algorithm too 
            method_name = methods[method] # figure out what's going on with 
            base = os.path.join('../../', 'dataset') 
            data = parse_dataset(base, dataset_name) 
            coarsen_data(data, dataset_name, method, method_name, coarse_levels, cpus, f'./dataset_coarsen_{int(coarse_levels * 100)}') 
