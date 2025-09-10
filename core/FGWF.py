import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import networkx as nx

from core.AlgOT import fgwd
from DataIO import StructuralDataSampler, process
from typing import List, Tuple
from GraphUtils import ensure_tensor, ensure_tensor_bulk, noisy_prob, A_to_G
from DataTools import MeasureNetwork


def inv_sigmoid(x, eps=1e-16):
    return torch.log((x + eps)/(1-x+eps))


def random_measure_network(num_nodes, params=(0.8, 0.2), prob=None, feat_dim=None):
    params = torch.Tensor(params)
    G = torch.multinomial(params, num_nodes**2, True)
    G = G.reshape(num_nodes, num_nodes)
    G = G - torch.diag(torch.diag(G))
    F = None if feat_dim is None else torch.zeros((num_nodes, feat_dim))
    m_net = MeasureNetwork(G, prob, F)
    return m_net


class FGWFModel(nn.Module):
    def __init__(self, m_nets=None, weights=None, alpha=0, atom_sizes=None, num_samples=None, 
                feat_dim=None,ot_layers=30, sinkhorn_layers=5, gwb_layers=5, tries=20, 
                multiplier=5, eps=1e-16):
        super().__init__()

        self.alpha = alpha
        self.ot_layers = ot_layers
        self.sinkhorn_layers = sinkhorn_layers
        self.gwb_layers = gwb_layers
        self.num_samples = num_samples
        self.atom_sizes = atom_sizes
        self.feat_dim = feat_dim
        self.tries = tries
        self.eps = eps

        self.softmax, self.sigmoid = nn.Softmax(dim=0), nn.Sigmoid()

        self.set_atoms(m_nets)
        self.set_weights(ensure_tensor(weights), multiplier)

    def set_atoms(self, m_nets):
        self.atoms, self.probs, self.feats = nn.ParameterList(), [], nn.ParameterList()

        if not (m_nets is None):
            self.atom_sizes = []
            for m_net in m_nets:
                G, P, F = m_net.get_all()
                pre_sig_atom = inv_sigmoid(G, self.eps)
                self.atoms.append(pre_sig_atom)
                self.probs.append(P)
                self.feats.append(F)
                self.atom_sizes.append(pre_sig_atom.shape[0])
                self.num_atoms = len(self.atom_sizes)

        else:
            assert(not (self.atom_sizes is None))
            for Nk in self.atom_sizes:
                G = torch.rand((Nk, Nk))
                P = torch.ones((Nk, 1))/Nk
                F = None if self.feat_dim is None else torch.rand((Nk, self.feat_dim))
                self.atoms.append(G)
                self.probs.append(P)
                self.feats.append(F)
                self.num_atoms = len(self.atoms)
        
    def set_weights(self, weights, multiplier=5):
        if not (weights is None):
            assert(weights.shape[0] == len(self.atoms))
            if (len(weights.shape) == 1) or weights.shape[1] == 0:
                weights = weights.reshape(-1, 1)

            self.weights = nn.Parameter(multiplier * weights)
        else:
            print(f"num_atoms num_samples = {self.num_atoms, self.num_samples}")
            assert(not (self.num_samples is None)) 
            rand_weights = torch.rand((self.num_atoms, self.num_samples))
            self.weights = nn.Parameter(rand_weights)

    def add_atom(self, m_net):
        atom, prob, feat = m_net.get_all()
        pre_sig_atom = inv_sigmoid(atom, self.eps)
        self.atoms.append(pre_sig_atom)
        self.probs.append(prob)
        self.feats.append(feat)
        self.num_atoms += 1

        weights = torch.zeros((self.num_atoms, self.num_samples))
        weights[:-1,:] = self.weights
        self.weights = nn.Parameter(weights)

    def output_weights(self, idx: int = None):
        if idx is None:
            return self.softmax(self.weights)
        else:
            return self.softmax(self.weights[:,idx])

    def output_atoms(self, idx: int = None):
        if idx is not None:
            return self.sigmoid(self.atoms[idx])
        else:
            return [self.sigmoid(self.atoms[idx]) for idx in range(len(self.atoms))]

    def output_measure_network(self, idx: int = None):
        if idx is not None:
            atom = self.output_atoms(idx)
            prob = self.probs[idx]
            feat = self.feats[idx]
            m_net = MeasureNetwork(atom, prob, feat)
            return m_net
        else:
            return [self.output_measure_network(i) for i in range(self.num_atoms)]

    def init_bary(self, M_init, num_nodes, rand_init_params=(0.8, 0.2)):
        if M_init is None:
            assert(not (num_nodes is None))
            M = random_measure_network(num_nodes, rand_init_params, None, self.feat_dim)
        
        else:
            assert(M_init.get_graph().shape[0] == M_init.get_prob().shape[0])
            M = M_init
    
        return M

    def comp_fgwb(self,
             pb: torch.Tensor,
             trans: List,
             weights: torch.Tensor) -> MeasureNetwork:
        """
        Solve GW Barycenter problem
        barycenter = argmin_{B} sum_k w[k] * d_gw(atom[k], B) via proximal point-based alternating optimization:

        step 1: Given current barycenter, for k = 1:K, we calculate trans[k] by the OT-PPA layer.
        step 2: Given new trans, we update barycenter by
            barycenter = sum_k trans[k] * atom[k] * trans[k]^T / (pb * pb^T)

        Args:
            pb: (nb, 1) vector (torch tensor), the empirical distribution of the nodes/samples of the barycenter
            trans: a dictionary {key: index of atoms, value: the (ns, nb) initial optimal transport}
            weights: (K,) vector (torch tensor), representing the weights of the atoms

        Returns:
            barycenter: (nb, nb) matrix (torch tensor) representing the updated GW barycenter
        """
        num_nodes = pb.shape[0]
        tmp1 = pb @ torch.t(pb)
        G = torch.zeros(num_nodes, num_nodes)
        compute_F = len(self.feats) > 0 and not (self.feats[0] is None) 
        
        if compute_F: 
            F = torch.zeros(num_nodes, self.feat_dim)
            tmp2 = pb @ torch.ones(1, self.feat_dim)

        for k in range(self.num_atoms): 
            Gk = self.output_atoms(k) 
            G += weights[k] * (trans[k].T @ Gk @ trans[k])
            if compute_F:
                F += weights[k] * (trans[k].T @ self.feats[k])

        G, F = G / tmp1, F / tmp2 if (compute_F) else None

        m_net = MeasureNetwork(G, pb, F)

        return m_net

    def fgwb(self, weights, M_init=None, num_nodes=None, noise_lvl=8, eps=1e-8, tries=20, rand_init_params=(0.8,0.2), verbose=False, keep_diag=True):
        assert((not M_init is None) or (not num_nodes is None))

        M_bary = self.init_bary(M_init, num_nodes, rand_init_params) 
        P_bary = M_bary.get_prob() # uniform if no pre-specified prob else prob

        for i in range(self.gwb_layers): 
            trans = []
            for k in range(self.num_atoms): 
                Mk = self.output_measure_network(k) 
                dk, Tk = fgwd(M1=Mk, M2=M_bary, 
                              alpha=self.alpha, 
                              ot_layers=self.ot_layers, 
                              sinkhorn_layers=self.sinkhorn_layers, 
                              noise_lvl=noise_lvl, eps=eps, tries=tries, 
                              verbose=verbose, sequence=None)
                if Tk is None:
                    print(f"Null transport plan at iter {i}, atom {k}")
                trans.append(Tk)
                if verbose:
                    print(f"iter {i}, atom {k}, dist = {dk}")
            M_bary = self.comp_fgwb(P_bary, trans, weights)
        
        if not keep_diag:
            mask = torch.ones_like(M_bary.graph) - torch.eye(M_bary.num_nodes)
            new_adj = M_bary.get_graph() * mask
            M_bary = MeasureNetwork(new_adj, M_bary.get_prob(), M_bary.get_feat())

        return M_bary

    def forward(self, M: MeasureNetwork, weights: torch.Tensor, noise_lvl=8, 
                eps=1e-10, tries=20,  verbose=False, init_mode=0):
        # if init_mode == 0: initialize barycenter to desired output, otherwise
        # initialize to random graph
        
        num_nodes = M.get_graph().shape[0]
        M_init = M if not init_mode else None

        M_bary = self.fgwb(weights, M_init, num_nodes=num_nodes, 
                           noise_lvl=noise_lvl, eps=eps, tries=tries, 
                           verbose=verbose)

        d_fgw, _ = fgwd(M, M_bary, alpha=self.alpha, ot_layers=self.ot_layers, 
                        sinkhorn_layers=self.sinkhorn_layers, noise_lvl=noise_lvl,
                        eps=eps, tries=tries, verbose=verbose, sequence=False)

        return d_fgw, M_bary


# update FGWF class to allow FGWF models with no atoms
class FGWF:
    """
    A simple PyTorch implementation of Fused Gromov-Wasserstein factorization model
    The feed-forward process imitates the proximal point algorithm or bregman admm
    """
    def __init__(self,
                 num_samples: int,
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
        """
        Args:
            num_samples: the number of samples
            size_atoms: a list, its length is the number of atoms, each element is the size of the corresponding atom
            feat_dim: the dimension of embedding
            ot_method: ppa or b-admm
            gamma: the weight of Bregman divergence term
            gwb_layers: the number of gwb layers in each gwf module
            ot_layers: the number of ot layers in each gwb module
        """
        super(FGWF, self).__init__()
        self.num_samples = num_samples
        self.feat_dim = feat_dim
        self.gwb_layers = gwb_layers
        self.ot_layers = ot_layers
        self.sinkhorn_layers = sinkhorn_layers
        self.alpha =  alpha
        self.cost_mode = (alpha != 0) or (feat_dim is None)
        self.noise_lvl = noise_lvl
        self.tries = tries
        self.eps = eps
        self.multiplier = multiplier

        self.set_model(prior_m_nets, prior_weights, num_samples, atom_sizes)

    def train(self,
              data, 
              epochs: int = 10, 
              lr_weights: float = 1e-1, 
              lr_atoms: float = 1e-2, 
              train_mode: int = 0, 
              bary_init_mode: int = 0,
              noise_lvl: float = 8.0, 
              tries: int = 20, 
              eps: float = 1e-8,
              sequence: bool = False,
              verbose=False):

        database, _ = process(data)

        w_idxs = [0] if train_mode == 2 else [] 
        atom_idxs = [i+1 for i in range(self.model.num_atoms)] if train_mode == 1 else [] 
        F_idxs = [i+self.model.num_atoms for i in atom_idxs] if train_mode == 1 else [] 

        inactive_params = w_idxs + atom_idxs + F_idxs 

        for i, param in enumerate(self.model.parameters()):
            if i in inactive_params:
                param.requires_grad = False

        optimizer_params = [{'params': self.model.weights},
                            {'params': self.model.atoms, 'lr': lr_atoms}]
        
        if not (self.model.feats[0] is None):
            for i in range(len(self.model.feats)-1):
                assert(self.model.feats[i].shape[1] == self.model.feats[i+1].shape[1])

            optimizer_params.append({'params': self.model.feats})

        optimizer = torch.optim.Adam(optimizer_params, 
                                lr=lr_weights)

        self.model.train() 

        data_sampler = StructuralDataSampler(database) 

        num_samples = len(data_sampler)
        index_samples = list(range(num_samples)) 
        best_loss, best_model = float("Inf"), None
        loss_series, bary_series, model_series = [], [], []

        for epoch in range(epochs): 
            optimizer.zero_grad()
            if sequence:
                loss_epoch_series, bary_epoch_series = [], []

            loss_epoch = 0

            for idx in index_samples: 
                m_net, w_idx = data_sampler[idx], self.model.output_weights(idx)


                d_fgw, M_bary = self.model(m_net, w_idx, noise_lvl, eps, 
                                               tries=tries, verbose=verbose, 
                                               init_mode=bary_init_mode)

                print(f"iter = {idx+1}/{num_samples}, weight = {np.round(w_idx.detach().numpy(), 2)}, loss = {np.round(d_fgw.detach().numpy(), 5)}")

                loss_epoch += d_fgw / num_samples

                if sequence:
                    loss_epoch_series.append(d_fgw.detach())
                    bary_epoch_series.append(M_bary)

                d_fgw.backward()
                optimizer.step()
                optimizer.zero_grad()


            if sequence:
                loss_series.append(loss_epoch_series)
                bary_series.append(bary_epoch_series)
                model_series.append(copy.deepcopy(self.model))

            if loss_epoch < best_loss:
                best_loss = loss_epoch
                best_model = model_series[-1] if sequence else copy.deepcopy(self.model)


        if not sequence:
            return best_model, best_loss 
        
        else:
            return best_model, best_loss, loss_series, bary_series, model_series

    def set_model(self, 
                  prior_m_nets=None, 
                  prior_weights=None,
                  num_samples=None, 
                  atom_sizes=None):
        
        assert(not (atom_sizes is None) or not (prior_m_nets is None))

        self.model = FGWFModel(prior_m_nets, 
                               prior_weights,
                               alpha=self.alpha, 
                               atom_sizes=atom_sizes, 
                               num_samples=num_samples, 
                               feat_dim=self.feat_dim, 
                               ot_layers=self.ot_layers, 
                               sinkhorn_layers=self.sinkhorn_layers, 
                               gwb_layers=self.gwb_layers, 
                               tries=self.tries, 
                               multiplier=self.multiplier,
                               eps=self.eps)
    
        self.num_atoms = self.model.num_atoms
        self.atom_sizes = self.model.atom_sizes

    def set_weights(self, prior_weights, multiplier=5):
        self.model.set_weights(prior_weights, multiplier=multiplier)

