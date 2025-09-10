import torch
import numpy as np
import torch.nn as nn
from OTLib import fgwd, fgwb
from GraphUtils import ensure_tensor, ensure_tensor_bulk
from typing import List, Tuple
from DataIO import StructuralDataSampler
import FGWF
from DataTools import MeasureNetwork
import matplotlib.pyplot as plt

class GWDict(FGWF.FGWF):
    def __init__(self, 
                #  num_samples: int,
                 prior_m_nets: List = None,
                 prior_weights=None,
                 atom_sizes: List = None,
                 feat_dim: int = 1,
                 gwb_layers: int = 5,
                 ot_layers: int = 30,
                 sinkhorn_layers: int=5,
                 alpha=0,
                 tries=20,
                 noise_lvl=8,
                 multiplier=5,
                 eps=1e-8): 
        
        num_samples = 1


        super().__init__(num_samples, prior_m_nets, prior_weights, atom_sizes, 
                         feat_dim, gwb_layers, ot_layers, sinkhorn_layers, alpha, 
                         tries, noise_lvl, multiplier, eps)
        

        self.fgwd = lambda X, Y: fgwd(X, Y, self.alpha, self.ot_layers, self.sinkhorn_layers, self.noise_lvl, self.eps, self.tries, False, False)[0]

    def fgwb(self, weights=None, bary_init=None, num_nodes=None, 
             rand_init_params=(0.8, 0.2), verbose=False):
        
        return self.model.fgwb(weights = ensure_tensor(weights), 
                               M_init=bary_init, 
                               num_nodes=num_nodes,
                               noise_lvl=self.noise_lvl,
                               eps=self.eps,
                               tries=self.tries,
                               rand_init_params=rand_init_params,
                               verbose=verbose)

    def closest_atom_weight(self, m_net, atoms):
        dists = torch.Tensor([self.fgwd(m_net, x) for x in atoms])
        best_idx = int(torch.argmin(dists))
        return torch.eye(len(atoms), requires_grad=True)[:,best_idx]

    def construct(self, m_nets, max_dist, lr=1, epochs=5, bary_init_mode=0, sequence=False, verbose=False):
        self.model.set_atoms([m_nets[0]])
        self.model.set_weights(torch.Tensor([[1.0]]))

        self.set_model(prior_m_nets=[m_nets[0]], 
                       prior_weights=[torch.ones(1,1)], 
                    #    num_samples=1, 
                       atom_sizes=None)

        train_mode, lr_atoms = 1, 0 # update weight, not atoms

        m_nets_dict = [m_nets[0]]
        weights_dict = [torch.Tensor([[1]])]

        for i, m_net in enumerate(m_nets[1:]):
            wi = self.closest_atom_weight(m_net, m_nets_dict)
            self.set_weights(wi, self.multiplier)

            res = self.train([m_net], 
                             epochs=epochs, 
                             lr_weights=lr, 
                             lr_atoms=lr_atoms, 
                             train_mode=train_mode, 
                             bary_init_mode=bary_init_mode, 
                             noise_lvl=self.noise_lvl, 
                             tries=self.tries, 
                             eps=self.eps, 
                             sequence=sequence, 
                             verbose=verbose) 
            
            best_model, best_loss = res[:2]
            wi = best_model.output_weights(0).detach().reshape(-1, 1)

            if best_loss < max_dist: 
                weights_dict.append(wi) 
            else: 
                m_nets_dict.append(m_net)
                wi = torch.eye(len(m_nets_dict))[:,-1].reshape(-1, 1)
                weights_dict.append(wi) 
                self.set_model(prior_m_nets=m_nets_dict, 
                               prior_weights=None, 
                               num_samples=1, 
                               atom_sizes=None)

        weights_dict_all = torch.zeros(len(m_nets_dict), len(m_nets))

        for i, wi in enumerate(weights_dict):
            Ni = wi.shape[0]
            print(i, Ni, wi.shape)
            weights_dict_all[:Ni,i] = wi[:,0]

        self.set_model(prior_m_nets = m_nets_dict, 
                       prior_weights = weights_dict_all, 
                       num_samples = 1)

    def construct_coarse(self, m_nets, coarse_lvls, max_dist=0.1, lr=1, epochs=5,
                          bary_init_mode=0, sequence=False, verbose=False):

        def get_best_weight(model, m_net):
            train_mode, lr_atoms = 1, 0.0
            res = model.train([m_net], 
                    epochs=epochs, 
                    lr_weights=lr, 
                    lr_atoms=lr_atoms, 
                    train_mode=train_mode, 
                    bary_init_mode=bary_init_mode, 
                    noise_lvl=model.noise_lvl, 
                    tries=model.tries, 
                    eps=model.eps, 
                    sequence=sequence, 
                    verbose=verbose) 

            best_model, best_loss = res[:2] 
            best_weight = best_model.output_weights(0).detach().reshape(-1, 1) 

            return best_loss.data, best_weight

        m_nets_dict, weights_dict = [], []

        for i, m_net in enumerate(m_nets):

            if len(m_nets_dict) > 0:
                wi = self.closest_atom_weight(m_net, m_nets_dict) 
                self.set_weights(wi, self.multiplier) 
                best_loss, wi = get_best_weight(self, m_net) 
            
            else:
                best_loss, wi = torch.inf, None

            if best_loss <= max_dist:
                print(f"Inside dict: {best_loss, max_dist}")
                weights_dict.append(wi)
                
            if best_loss > max_dist:
                print(f"Outside dict: {best_loss, max_dist}")
                lvls = torch.Tensor(coarse_lvls + [m_net.num_nodes])
                sorted_lvls = lvls.sort(descending=True)[0]
                sorted_lvls = sorted_lvls.numpy()
                best_w, best_mnet_coarse = None, None
                
                for nk in sorted_lvls:
                    if nk == m_net.num_nodes:
                        m_net_coarse_k = m_net
                    else:
                        m_net_coarse_k = m_net.coarsen(nk, 0)
                    new_m_nets = m_nets_dict + [m_net_coarse_k]
                    prior_weights = torch.eye(len(new_m_nets))[:,-1]

                    # m_net_coarse_k.plot()
                    # plt.show()

                    model_k = GWDict(prior_m_nets=new_m_nets, 
                                     prior_weights=prior_weights, 
                                     atom_sizes=None, 
                                     feat_dim=self.feat_dim, 
                                     gwb_layers=self.gwb_layers, 
                                     ot_layers=self.ot_layers, 
                                     sinkhorn_layers=self.sinkhorn_layers, 
                                     alpha=self.alpha, 
                                     tries=self.tries, 
                                     noise_lvl=self.noise_lvl, 
                                     multiplier=self.multiplier, 
                                     eps=self.eps)      
                    
                    best_loss_k, wk = get_best_weight(model_k, m_net)  
                    print(f"coarse level = {nk}, d_ij = {best_loss_k}") 


                    if best_w == None or (best_loss_k <= max_dist):
                        best_w, best_mnet_coarse = wk, m_net_coarse_k
                
                print(f"Added atom with {best_mnet_coarse.num_nodes} nodes\n")
                weights_dict.append(best_w)
                m_nets_dict.append(best_mnet_coarse) 

                self.set_model(prior_m_nets=m_nets_dict, 
                               prior_weights=None, 
                               num_samples=1, 
                               atom_sizes=None)
            
            print('-'*200 + "\n")
            
        weights_dict_all = torch.zeros(len(m_nets_dict), len(m_nets))

        for i, wi in enumerate(weights_dict):
            Ni = wi.shape[0]
            weights_dict_all[:Ni,i] = wi[:,0]

        self.set_model(prior_m_nets = m_nets_dict, 
                       prior_weights = weights_dict_all, 
                       num_samples = weights_dict_all.shape[0])                        
