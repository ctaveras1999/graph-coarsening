import sys
sys.path.insert(0, './core')
sys.path.insert(0, '')
# from DataTools import MeasureNetwork
from tqdm import tqdm
from CoarsenTools import coarsen_data, parse_dataset
import os
import torch
import numpy as np

if __name__ == "__main__":
    datasets = ['MSRC_9', 'MUTAG', 'PTC_MR', 'IMDB-BINARY', 'ENZYMES']
    methods =  ["MGC", "SGC", "WGC", "KGPC", "GPC"] # STATIC -- Don't Change!
    dir_prefix = 'coarsened_final'
    coarse_levels = list(np.linspace(0, 1, 21)[-2:0:-1]) 

    cpus = 60
    for dataset_name in [datasets[3]]:
        for method in tqdm([3, 4]):
            base = os.path.join('.', 'dataset') 
            data = parse_dataset(base, dataset_name) 
            coarsen_data(data, dataset_name, method, coarse_levels, cpus, dir_prefix) 
