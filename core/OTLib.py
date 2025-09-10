from core.AlgOT import fgwd
from core.GraphUtils import ensure_tensor
from core.FGWF import FGWFModel

def fgwb(m_nets, w=None, n_output=None, M_init=None, alpha=0, noise_lvl=7, ot_layers=40, 
         sinkhorn_layers=5, gwb_layers=5, eps=1e-10, tries=20, 
         rand_init_params=(0.8, 0.2), verbose=False):
    
    num_graphs = len(m_nets)

    if w is None:
        w = [1/num_graphs for _ in range(num_graphs)]

    weight = ensure_tensor(w)

    for i in range(len(m_nets)-1):
        curr_dim, next_dim = m_nets[i].feat_dim, m_nets[i+1].feat_dim
        assert(curr_dim == next_dim)

    feat_dim, num_samples = m_nets[0].feat_dim, 1

    model = FGWFModel(m_nets=m_nets, weights=weight, alpha=alpha, num_samples=num_samples, 
                      feat_dim=feat_dim, ot_layers=ot_layers, sinkhorn_layers=sinkhorn_layers, 
                      gwb_layers=gwb_layers, tries=tries, multiplier=10, eps=eps)

    M_bary = model.fgwb(w, M_init, n_output, noise_lvl, eps, tries, 
                        rand_init_params, verbose=verbose)
    
    return M_bary