
"""
In this document we modify the method presented in Jiang et al's paper by 
computing the transport maps using the proximal method as detailed in Xu's paper. 
"""


import sys 
sys.path.insert(1, '../')

import torch
import torch.nn as nn
from DataTools import MeasureNetwork
from OTLib import fgwb


class MLPcondensed(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput, mlp_type):
        super().__init__()
        final_layer = nn.Sigmoid if mlp_type == 0 else nn.Softmax
        self.layers = nn.Sequential(
            nn.Linear(nInput, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nOutput),
            final_layer()
        )

    def forward(self, x):
        x = x.reshape(-1)
        # print(x.shape)
        return(self.layers(x))

class AutoReg(nn.Module):
    def __init__(self, K, Nt, num_hidden):
        super().__init__()
        self.K = K
        self.coeffs = torch.abs(torch.rand(K))
        self.coeffs = self.coeffs / torch.sum(self.coeffs)
        self.g1 = MLPcondensed(Nt * Nt * self.K, num_hidden, Nt ** 2, 0)#.reshape(Nt, Nt)

    def align(self, Gs, Nt, alpha):
        assert(len(Gs) == self.K) 
        w = torch.ones(self.K) / self.K 
        M_bary = fgwb(Gs, w, Nt, None, alpha, noise_lvl=6)
        mats, feats = [], [] 
        mean = torch.zeros((Nt,Nt))
        for G in Gs: 
            T = M_bary.comp_dist(G)[1] 
            mat_aligned = T @ G.graph @ T.T * (Nt ** 2)
            mean += mat_aligned / Nt
            feat_aligned = T @ G.feat 
            mats.append(mat_aligned) 
            feats.append(feat_aligned) 
        return mats, feats 

    def forward(self, Gs, Nt, alpha=0): 
        # compute barycenter -> compute transport maps from graph to barycenter
        # use transport maps to align all graphs -> concatenate aligned graphs 
        # input of neural network is of size: Nt x Nt x K
        # output of neural network is of size: Nt x Nt 
        mats, feats = self.align(Gs, Nt, alpha) 
        prob_mat = self.g1(torch.cat(mats)).reshape(Nt, Nt)
        return MeasureNetwork(prob_mat, grad=True)#, pred_feat

