import torch.nn as nn
import torch.nn.functional as fun
import ot
from torch.optim import Adam
import copy
from OTLib import fgwb, fgwd
from DataTools import MeasureNetwork
import numpy as np
import torch

class ModelAlpha(nn.Module):

    def __init__(self, n_template, n_params=50):
        super(ModelAlpha, self).__init__()
        self.fc1 = nn.Linear(1, n_params)  # 5*5 from image dimension
        self.fc2 = nn.Linear(n_params, n_params)
        self.fc3 = nn.Linear(n_params, n_template)

    def forward(self, x):
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = fun.softmax(self.fc3(x), dim=-1)
        return x

class DeepFgwEstimator:
    def __init__(self, n_templates, n_params=20):

        # FGW parameter
        self.alpha = 0
        self.max_iter = 5

        # Model parameter
        self.n_templates = n_templates
        self.weights = ModelAlpha(n_template=self.n_templates, n_params=n_params)
        self.nb_node_template = torch.arange(1, n_templates)
        self.feature_dim = 1
        self.C_templates = None 
        self.F_templates = None 
        self.P_templates = None 
        self.params = None 
        self.n_out = 40 

        # Training parameter 
        self.n_epochs = 5 
        self.lr = 0.01 

    def loss(self, Y_true, Y_preds, compute_both=False): 
        # Choose distribution over nodes: uniform 
        if compute_both:
            Y_preds = Y_preds
        else:
            Y_preds = [Y_preds]

        num_outputs = 2 if compute_both else 1
        best_dist = torch.zeros(num_outputs)

        for k, Y_pred in enumerate(Y_preds):
            C_true, P_true, F_true = Y_true.get_all() 
            C_pred, P_pred, F_pred = Y_pred.get_all() 
            P_true, P_pred = P_true.reshape(-1), P_pred.reshape(-1) 

            # Compute euclidean distance matrix between F_pred and F_true
            n_u = torch.linalg.norm(F_pred, axis=1).reshape((-1, 1)) ** 2 
            n_v = torch.linalg.norm(F_true, axis=1).reshape((-1, 1)) ** 2 
            n_uv = n_u + n_v.T 
            B = torch.mm(F_pred, F_true.T) 
            M = n_uv - 2 * B 

            # Compute FGW distance
            fgw1 = ot.gromov.fused_gromov_wasserstein2(M, 
                                                    C_pred, C_true, 
                                                    P_pred, P_true, 
                                                    loss_fun='square_loss', 
                                                    alpha=1-self.alpha,
                                                    log=False)
            
            fgw2 = fgwd(Y_true, Y_pred, self.alpha, noise_lvl=7)[0] 

            best_dist[k] = torch.min(fgw1, fgw2) 
                
        return torch.min(best_dist)

    def train(self, X, Y, dict_learning=False, Y_templates=None, compute_both=False):

        if Y_templates is None: 
            if self.C_templates is None: 
                self.C_templates = [] 
                self.P_templates = [] 
                self.F_templates = [] 
                for i in range(self.n_templates):
                    if isinstance(self.nb_node_template, int):
                        Ni = self.nb_node_template
                    else:
                        Ni = self.nb_node_template[i]
                    C = torch.rand(Ni, Ni, requires_grad=dict_learning)
                    F = torch.rand(Ni, self.feature_dim, dtype=torch.float32, requires_grad=dict_learning)
                    P = torch.ones(Ni)/Ni
                    self.C_templates.append(C)
                    self.P_templates.append(P)
                    self.F_templates.append(F)

        # Initialize templates 
        else:
            self.C_templates, self.P_templates, self.F_templates = [], [], []

            for m_net in Y_templates:
                x,y,z = m_net.get_all()
                x_grad = x.clone().detach().requires_grad_(True)
                z_grad = z.clone().detach().requires_grad_(True)

                self.C_templates.append(x_grad)
                self.P_templates.append(y)
                self.F_templates.append(z_grad)

        # Define model parameters for gradient descent
        print(f"dict learning = {dict_learning}")
        if dict_learning:
            self.params = [*self.weights.parameters(), *self.C_templates, *self.F_templates]#, *self.P_templates]
        else:
            self.params = [*self.weights.parameters()]

        # Define torch optimizer
        optimizer = Adam(params=self.params, lr=self.lr)

        # Gradient descent
        N = X.shape[0]
        loss_iter = []
        loss_iter_all = []
        models_all = []
        for e in range(self.n_epochs):

            # One epoch
            loss_e = 0
            for i in range(N):
                res_pred = self.predict(X[i], compute_both=compute_both)
                loss_i = self.loss(Y[i], res_pred, compute_both=compute_both)

                loss_to_print = np.round(loss_i.detach().numpy() * 1e3) / 1e3

                loss_i.backward() 
                optimizer.step() 
                optimizer.zero_grad() 

                # Clamping C in [0,1]
                with torch.no_grad():
                    for C in self.C_templates:
                        C[:] = C.clamp(0, 1)

                print(f"loss = {loss_to_print}")
                
                # Gradient step
                loss_iter_all.append(float(loss_i.detach().cpu().numpy()))

                with torch.no_grad():
                    for C in self.C_templates:
                        C[:] = C.clamp(0, 1)

                loss_e = loss_i + loss_e 
            

            models_all.append(copy.deepcopy(self.weights))
            loss_iter.append(float(loss_e.detach().cpu().numpy()) / N)
            print(f"Iter {e} = {np.round(loss_iter[-1], 3)}")
            print("="*150)


        return loss_iter, (loss_iter_all, models_all)

    def predict(self, x_te, p=None, compute_both=False):
        # Predict weights
        lambdas = self.weights(x_te)[0]
        print(f"w({np.round(x_te.reshape(-1).detach().numpy(), 1)[0]}) = {np.round(lambdas.detach().numpy(), 2)}", end=" ")

        # Compute barycenter from weights and templates
        probs = [x.reshape(-1) for x in self.P_templates]

        F_bary, C_bary = ot.gromov.fgw_barycenters(N=self.n_out, 
                                                   Ys=self.F_templates, 
                                                   Cs=self.C_templates, 
                                                   ps=probs,
                                                   p=p,
                                                   lambdas=lambdas, 
                                                   alpha=1-self.alpha, 
                                                   loss_fun='square_loss',
                                                   max_iter=self.max_iter, 
                                                   tol=1e-9)

        bary1 = MeasureNetwork(C_bary, p, F_bary)

        if not compute_both:
            return bary1

        else:
            m_nets = [MeasureNetwork(x,y,z) for (x,y,z) in zip(self.C_templates, self.P_templates, self.F_templates)]
            bary2 = fgwb(m_nets, lambdas, self.n_out, bary1, self.alpha, 7, 30, 5, 10, 1e-10, tries=5)
        return bary1, bary2
