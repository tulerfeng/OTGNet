import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
import torch.optim as optim



class TimeEncoder(torch.nn.Module):
    def __init__(self, expand_dim=2, device=None):
        super(TimeEncoder, self).__init__()
        
        time_dim = expand_dim
        self.device = device
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()).to(device)
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float()).to(device)
    
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic 
    