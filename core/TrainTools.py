import os
import sys
sys.path.insert(1, "../")

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import torch
import FusedGromovWassersteinFactorization as FGWF
from DataUtils import *
import pandas as pd
from AlgOT import ot_fgw
from tqdm import tqdm
import copy
from DataIO import process
from tqdm import tqdm
import pickle 
from VisualizationTools import transform_weights
from multiprocessing import Pool

def dummy_model(atoms, weight_prior=None, gamma=5e-2, gwb_iter=20, ot_iter=20, alpha=0, cost_mode=0):
    atom_prior = [torch.Tensor(G_to_A(x)) for x in atoms]
    size_atoms = [x.shape[0] for x in atom_prior]
    num_atoms = len(atoms)
    num_samples = len(atoms)
    num_classes, dim_embedding = 2, 1

    gamma=5e-2
    gwb_iter=20
    ot_iter=20
    alpha=0
    dw=dw_type=dbasis=entw=sparsew=0

    model = FGWF.FGWF(num_samples=num_samples, 
                        num_classes=num_classes,
                        size_atoms=size_atoms,
                        dim_embedding=dim_embedding,
                        ot_method='ppa',
                        gamma=gamma,
                        gwb_layers=gwb_iter,
                        ot_layers=ot_iter,
                        atom_prior=atom_prior,
                        weight_prior=weight_prior,
                        alpha=alpha,
                        weight_diff = dw,
                        weight_diff_type = dw_type,
                        basis_diff = dbasis, 
                        weight_entropy = entw,
                        cost_mode=cost_mode, 
                        sparsity_factor = sparsew)

    return model

def train_real(fields, size_atoms, cost_mode, alpha=0, lr_atoms=5e-2, lr_weights=1e-1, epochs=20, gamma=5e-2, gwb_iter=20, ot_iter=20, weight_decay=0, gw_type=0, dw_type=1, dw=0, dbasis=0, entw=0, sparsew=0, save_loss=True, save_weights=True, save_atoms=True, size_batch=1, train_mode=0, atom_prior=None, weight_prior=None):
    num_atoms = len(size_atoms)
    if not (atom_prior is None):
        num_atoms = len(atom_prior)
        size_atoms = [x.shape for x in atom_prior]
    graph_seq = fields["Gs"]
    data, sampler = process(graph_seq, num_atoms)
    # weight_decay = 0
    num_classes, dim_embedding = 2, 1
    res_dir = fields["name"]
    model = FGWF.FGWF(num_samples=len(data), 
                      num_classes=num_classes,
                      size_atoms=size_atoms,
                      dim_embedding=dim_embedding,
                      ot_method='ppa',
                      gamma=gamma,
                      gwb_layers=gwb_iter,
                      ot_layers=ot_iter,
                      atom_prior=atom_prior,
                      weight_prior=weight_prior,
                      alpha=alpha,
                      weight_diff = dw,
                      weight_diff_type = dw_type,
                      basis_diff = dbasis, 
                      weight_entropy = entw,
                      cost_mode=cost_mode, 
                      sparsity_factor=sparsew)
    
    if not os.path.isdir('results'):
        os.mkdir('./results')

    res_dir_full = os.path.join('results', res_dir)

    if not os.path.isdir(res_dir_full):
        os.mkdir(res_dir_full)

    subfolders = [f.path for f in os.scandir(res_dir_full) if f.is_dir() and ('run' in f.path)]
    num_runs = [int(x.split('_')[-1]) for x in subfolders if 'run' in x]
    this_run = 1 if len(num_runs) == 0 else max(num_runs) + 1
    
    path_dir = os.path.join(res_dir_full, f'run_{this_run}')    
    os.mkdir(path_dir)

    atom_dir =  os.path.join(path_dir, 'atoms_viz')
    model_dir = os.path.join(path_dir, 'models')
    graph_dir = os.path.join(path_dir, 'graphs') 
    loss_dir = os.path.join(path_dir, 'loss')
    weights_dir = os.path.join(path_dir, 'weights')
    atoms_dir = os.path.join(path_dir, 'atoms')

    os.mkdir(atom_dir)
    os.mkdir(model_dir)
    if save_loss:
        os.mkdir(loss_dir)

    if save_weights:
        os.mkdir(weights_dir)

    if save_atoms:
        os.mkdir(atoms_dir)

    fields_pkl = os.path.join(path_dir, 'fields.pkl')

    with open(fields_pkl, 'wb') as f:
        pickle.dump(fields, f)

    viz_prefix = os.path.join(atom_dir,    'atom_viz')
    model_prefix = os.path.join(model_dir, 'model')
    graph_prefix = os.path.join(graph_dir, 'graph')
    loss_prefix = os.path.join(loss_dir,   'loss')
    weights_prefix = os.path.join(weights_dir, 'weights')
    atoms_prefix = os.path.join(atoms_dir, 'atoms')

    model, int_models = FGWF.train_usl(model, 
                                        data,
                                        size_batch=size_batch,
                                        epochs=epochs,
                                        lr_atoms=lr_atoms,
                                        lr_weights=lr_weights,
                                        weight_decay=weight_decay,
                                        mode=train_mode,
                                        gw_type=gw_type,
                                        visualize_prefix=viz_prefix, 
                                        model_prefix=model_prefix,
                                        weights_prefix=weights_prefix,
                                        atoms_prefix=atoms_prefix,
                                        int_models=True,
                                        loss_prefix=loss_prefix,
                                        save_loss=save_loss, 
                                        save_weights=save_weights, 
                                        save_atoms=save_atoms)

    return model, int_models

def predict(model, sampler, bary_init_mode=0):
    predictions = []
    predictions_full = []
    for k in range(len(sampler)): 
        Gk_init = sampler[k][0] if bary_init_mode == 0 else torch.rand(sampler[k][0].shape)
        num_non_zero = (sampler[k][0].ravel() > 0).sum()
        wk =  model.output_weights(k).detach()
        Gk_hat = FGWF.gen_graph(model, Gk_init, wk, to_np=True)
        predictions_full.append(Gk_hat)
        largest_idx = np.argsort(Gk_hat.ravel())[-num_non_zero-1:]
        new_graph = np.zeros_like(Gk_hat)
        for idx in largest_idx:
            row_idx, col_idx = idx // Gk_init.shape[0], idx % Gk_init.shape[0]
            new_graph[row_idx, col_idx] = Gk_hat[row_idx, col_idx]# np.clip(Gk_hat[row_idx, col_idx], 0, 1)
            new_graph[col_idx, row_idx] = Gk_hat[col_idx, row_idx]# np.clip(Gk_hat[col_idx, row_idx], 0, 1)
        Gk_hat = new_graph
        predictions.append(Gk_hat)
    return predictions, predictions_full

def evaluate(model, sampler, predictions):
    err, N = 0, len(sampler)
    dists = np.zeros(N)
    for k, (X, X_hat) in enumerate(zip([x[0] for x in sampler], predictions)):
        if len(X) == len(X_hat):
            p_s = p_t = noisy_unif(len(X))
        else:
            p_s, p_t = noisy_unif(len(X)), noisy_unif(len(X_hat))
        X, X_hat = torch.Tensor(X), torch.Tensor(X_hat)
        feat_s = feat_t = None 
        d_gw = ot_fgw(X, X_hat, p_s, p_t, 'ppa', model.gamma, model.ot_layers, 10, feat_s, feat_t, model.alpha, model.cost_mode, 0)[0].detach()
        err += d_gw
        dists[k] = d_gw.numpy()
    avg_err = err / N
    return avg_err, dists

def atom_span_plots(dists, weights):
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    colors = ["red", "green", "blue", "k"]
    for i in range(dists.shape[-1]):
        axs[0].plot(np.mean(dists[:,:,i], axis=0), c=colors[i], linewidth=1)
        axs[0].set_title("Distance to atoms")
        # if i < 3:
        #     axs[1].plot(weights[i,:], c=colors[i])
        #     axs[1].set_title("Atom Weight")
    plt.tight_layout()
    axs[0].legend(["B1", "B2", "B3", "Init"])
    transform_weights(weights.T, dists[0,:,:-1], axs[1])
    plt.show()

def atom_span_test(model, num_inits, num_samples, init_type=0):
    if init_type == 1:
        weights = model.output_weights().detach()
        num_samples = max(weights.shape)
    else:
        weights = simplex_sample(num_samples, model.num_atoms)
    dists = np.zeros((num_inits, num_samples, model.num_atoms + 1))
    # weights = simplex_sample(num_samples, model.num_atoms)
    inits = [torch.rand(model.atoms[0].shape) for _ in range(num_inits)]
    atoms = [x.detach() for x in model.output_atoms()]
    barycenters = []
    for i, Gi in enumerate(inits):
        G_hats = []
        for j, wj in tqdm(enumerate(weights.T)):
            G_hat = FGWF.gen_graph(model, Gi, wj, False)
            G_hats.append(G_hat)
            for k, Bk in enumerate(atoms + [Gi]):
                p1, p2 = noisy_unif(G_hat.shape[0]), noisy_unif(Bk.shape[0])
                dists[i,j,k] = ot_fgw(G_hat, Bk, p1, p2, 'ppa', model.gamma, model.ot_layers, 10, None, None, model.alpha, model.cost_mode, 0)[0].detach()
        barycenters.append(G_hats)
    
    from itertools import combinations
    idxs = list(combinations(range(model.num_atoms), 2))

    atom_dists = []
    num_combos = len(idxs)
    for (i, j) in idxs: 
        x, y = model.output_atoms(i), model.output_atoms(j)
        px, py = noisy_unif(x.shape[0]), noisy_unif(y.shape[0])
        embx, emby = model.embeddings[i].data, model.embeddings[j].data
        dist_xy = ot_fgw(x, y, px, py, model.ot_method, model.gamma, model.ot_layers, emb_s=embx, emb_t=emby, alpha=model.alpha, cost_mode=model.cost_mode)[0]
        dist_xy /= num_combos 
        atom_dists.append(dist_xy.detach().numpy())

    atom_span_plots(dists, weights)

    # _ = transform_weights(weights.T, dists[0,:,:-1])

    return dists, G_hats, weights, barycenters, atom_dists

def train_models(fields, size_atoms, cost_mode, reg_pairs, alpha=0, lr_atoms=5e-2, lr_weights=1e-1, epochs=20, gamma=5e-2, gwb_iter=20, ot_iter=20, train_mode=0, atom_prior=None, weight_prior=None):               
    pool = Pool(processes=6)
    #          fields, size_atoms, cost_mode, alpha=0, lr=5e-2, epochs=20, gamma=5e-2, gwb_iter=20, ot_iter=20, dw_type=1, dw=0, dbasis=0, entw=0, sparsew=0, save_loss=False, size_batch=1
    inputs = [(fields, size_atoms, cost_mode, alpha, lr_atoms, lr_weights, epochs, gamma, gwb_iter, ot_iter, *x, True, True, True, 1, train_mode, atom_prior, weight_prior) for x in reg_pairs]
    outputs = pool.starmap(train_real, inputs)
    pool.close()
    pool.terminate()
    pool.join()
    return outputs

def get_model(res_dir, run, pre_path=""):
    if pre_path == "":
        pre_dir = 'results'
    else:
        pre_dir = os.path.join(pre_path, 'results')
    res_dir_full = os.path.join(pre_dir, res_dir)
    path_dir = os.path.join(res_dir_full, f'run_{run}')    
    models_dir = os.path.join(path_dir, 'models')
    fields_pkl = os.path.join(path_dir, 'fields.pkl')
    with open(fields_pkl, 'rb') as f:
        fields = pickle.load(f)

    models = []
    model_paths = os.listdir(models_dir)
    model_paths_sorted_idx = np.argsort([int(x.split("_")[-1][:-4]) for x in model_paths])
    model_paths = [model_paths[i] for i in model_paths_sorted_idx]

    for x in model_paths:    
        model_i = torch.load(os.path.join(models_dir, x))
        models.append(model_i)

    model = copy.deepcopy(models[-1])

    return models, model, fields

def get_losses(name, run, pre_path=""):
    if pre_path == "":
        loss_path = f"./results/{name}/run_{run}/loss/"
    else: 
        loss_path = os.path.join(pre_path, f"results/{name}/run_{run}/loss/")
    loss_paths = os.listdir(loss_path)
    loss_paths_sorted_idx = np.argsort([int(x.split("_")[-1][:-4]) for x in loss_paths])
    loss_paths = [loss_paths[i] for i in loss_paths_sorted_idx]
    losses = [np.load(os.path.join(loss_path, x)).T for x in loss_paths]
    return losses

def get_weights(name, run, pre_path = ""):
    if pre_path == "":
        w_path = f"./results/{name}/run_{run}/weights/"
    else:
        w_path = os.path.join(pre_path, f"results/{name}/run_{run}/weights/")
    w_paths = os.listdir(w_path)
    w_paths_sorted_idx = np.argsort([int(x.split("_")[-1][:-4]) for x in w_paths])
    w_paths = [w_paths[i] for i in w_paths_sorted_idx]
    losses = [np.load(os.path.join(w_path, x)) for x in w_paths]
    return losses

def get_atoms(name, run, pre_path = ""):
    if pre_path == "":
        w_path = f"./results/{name}/run_{run}/atoms/"
    else:
        w_path = os.path.join(pre_path, "results/{name}/run_{run}/atoms/")
    w_paths = os.listdir(w_path)
    w_paths_sorted_idx = np.argsort([int(x.split("_")[-1][:-4]) for x in w_paths])
    w_paths = [w_paths[i] for i in w_paths_sorted_idx]
    losses = [np.load(os.path.join(w_path, x), allow_pickle=True) for x in w_paths]
    return losses

def display_atoms(model, fields, axs=None, artificial=False, fix_pos=True, figsize=(10,5)):
    atoms = [A_to_G(x.detach().numpy()) for x in model.output_atoms()]
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    n_atoms = len(atoms)
    plt.figure(figsize=figsize)
    if axs is None and not artificial:
        ax1 = plt.subplot(1,n_atoms,1)
        ax2 = plt.subplot(1,n_atoms,2)
        ax3 = plt.subplot(1,n_atoms,3)
        axs = [ax1, ax2, ax3]

    elif axs is None and not artificial:
        ax1 = plt.subplot(2,n_atoms,1)
        ax2 = plt.subplot(2,n_atoms,2)
        ax3 = plt.subplot(2,n_atoms,3)
        ax4 = plt.subplot(2,n_atoms,1)
        ax5 = plt.subplot(2,n_atoms,2)
        ax6 = plt.subplot(2,n_atoms,3)
        axs = [[ax1, ax2, ax3], [ax4, ax5, ax6]]

    if fix_pos:
        sparsest_atom = np.argmin([x.sum().detach().numpy() for x in model.output_atoms()])
        pos_hat = nx.spring_layout(atoms[sparsest_atom])

    for i, atom_G in enumerate(atoms):
        pos_hat = pos_hat if fix_pos else nx.spring_layout(atom_G)
        widths = [w for (*edge, w) in atom_G.edges.data('weight')]

        if not artificial:
            nx.draw(atom_G, node_size=20, with_labels=False, node_color=colors[i], alpha=1, width=widths, pos=pos_hat, ax=axs[i])
            axs[i].set_title(f"Atom {i+1}")
        if artificial:
            nx.draw(atom_G, node_size=20, with_labels=False, node_color=colors[i], alpha=1, width=widths, pos=pos_hat, ax=axs[0][i])
            Bi = fields["basis"][i]
            pos = nx.spring_layout(Bi)
            B_widths = [w for (*edge, w) in Bi.edges.data('weight')]
            nx.draw(Bi, node_size=20, with_labels=False, node_color='k', alpha=1, width=B_widths, pos=pos, ax=axs[1][i])
            axs[0, i].set_title(f"Atom {i+1}")
    return axs

def display_results(model, fields, fix_pos=True):
    # Show learned atoms
    artificial = fields["name"] == "artificial"

    plt.figure(figsize=(10,10))
    n_atoms = model.num_atoms
    if not artificial:
        ax1 = plt.subplot(3,n_atoms,1)
        ax2 = plt.subplot(3,n_atoms,2)
        ax3 = plt.subplot(3,n_atoms,3)
        axs = [ax1, ax2, ax3]
        ax_bot = plt.subplot(3,1,2)
        ax_simp = plt.subplot(3,1,3)
    else:
        ax1 = plt.subplot(4,n_atoms,1)
        ax2 = plt.subplot(4,n_atoms,2)
        ax3 = plt.subplot(4,n_atoms,3)
        ax4 = plt.subplot(4,n_atoms,1)
        ax5 = plt.subplot(4,n_atoms,2)
        ax6 = plt.subplot(4,n_atoms,3)
        axs = [[ax1, ax2, ax3], [ax4, ax5, ax6]]
        ax_bot = plt.subplot(4,1,3)
        ax_simp = plt.subplot(4,1,4)

    display_atoms(model, fields, axs, artificial, fix_pos)


    # Compute weight per channel
    weights = model.output_weights().detach().numpy().T
    # plt.figure(figsize=(6,3))
    ax_bot.plot(weights[:,0], c='red')
    ax_bot.plot(weights[:,1], c='green')
    ax_bot.plot(weights[:,2], c='blue')
    ax_bot.legend(["Atom 1", "Atom 2", "Atom 3"])
    ax_bot.set_title("Channel weights over time")
    ax_bot.set_xlabel("n")
    ax_bot.set_ylabel("Channel weight")
    weight_avgs = np.mean(weights, axis=0)
    print("Average Weight per Channel:", weight_avgs)

    # Compute distance between atoms
    idxs = list(combinations(list(range(model.num_atoms)), 2))
    num_combos = len(idxs)
    atom_dists = []
    for (i, j) in idxs: 
        x, y = model.output_atoms(i), model.output_atoms(j)
        if x.shape[0] != y.shape[0]:
            px, py = noisy_unif(x.shape[0]), noisy_unif(y.shape[0])
        else:
            px = py = noisy_unif(x.shape[0])

        embx, emby = model.embeddings[i].data, model.embeddings[j].data
        dist_xy = ot_fgw(x, y, px, px, model.ot_method, model.gamma, model.ot_layers, emb_s=embx, emb_t=emby, alpha=model.alpha, cost_mode=model.cost_mode)[0]
        dist_xy /= num_combos 
        atom_dists.append(dist_xy.detach().numpy())

    res_str = ""
    for k, ((i,j), d_ij) in enumerate(zip(idxs, atom_dists)):
        round_num = 4
        d_ij_round = int(d_ij * 10**round_num) / 10**round_num
        sep = ", " if k + 1 != len(atom_dists) else ""
        res_str += f"d(B{i+1}, B{j+1}) = {d_ij_round}" + sep
    print(res_str)

    dists = np.zeros((model.weights.shape[-1], model.num_atoms))
    # Gi = torch.rand(model.atoms[0].shape)
    G_hats = []
    for j, wj in tqdm(enumerate(weights)):
        # G_hat = FGWF.gen_graph(model, Gi, wj, False)
        Gj = torch.Tensor(G_to_A(fields["Gs"][j]))
        G_hat = FGWF.gen_graph(model, Gj, wj, False)
        G_hats.append(G_hat)
        for k, Bk in enumerate([x.detach() for x in model.output_atoms()]):
            p1, p2 = noisy_unif(G_hat.shape[0]), noisy_unif(Bk.shape[0])
            if G_hat.shape[0] == Bk.shape[0]:
                p2 = p1
            dists[j,k] = ot_fgw(G_hat, Bk, p1, p2, 'ppa', model.gamma, model.ot_layers, 10, None, None, model.alpha, model.cost_mode, 0)[0].detach()

    _ = transform_weights(weights, dists, ax_simp)

    plt.show()
    return atom_dists, dists, G_hats

