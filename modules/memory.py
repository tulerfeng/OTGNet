import torch
import numpy as np
from torch import nn


class Memory(nn.Module):
  def __init__(self, n_nodes, emb_dim, device='cpu', features=None):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.emb_dim = emb_dim
    self.device = device
    self.features=features
    self.__init_memory__()

  def __init_memory__(self, nodes=None, seed=0):
    seed=0
    torch.manual_seed(seed)
    if nodes is not None:
        tmp = nn.Parameter(torch.zeros((self.n_nodes+1, self.emb_dim)).to(self.device), requires_grad=False)
        nn.init.xavier_normal_(tmp)
        self.emb.detach_()
        self.emb[list(nodes)] = tmp[list(nodes)]
    else:    
        if self.features is None:
            self.emb = nn.Parameter(torch.zeros((self.n_nodes+1, self.emb_dim)).to(self.device), requires_grad=False)
            nn.init.xavier_normal_(self.emb)
            self.emb[0] = 0.0
        else:
            self.emb = nn.Parameter(torch.zeros((self.features.shape[0]+1,self.features.shape[1])).float().to(self.device), requires_grad=False)
            self.emb[0] = 0.0
            self.emb[1:] = nn.Parameter(torch.tensor(self.features).float().to(self.device), requires_grad=False) 

  def detach_memory(self):
    self.emb.detach_()
