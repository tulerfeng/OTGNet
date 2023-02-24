
import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
import torch.optim as optim
import math




class IB(nn.Module): 
    def __init__(self, shape_x, shape_z, shape_y, label_src, label_dst, per_class, device, beta=0.3, dis_IB=False, ch_IB='m', n_task=6, lr=0.005):
        super(IB, self).__init__()
        self.label_src = label_src
        self.label_dst = label_dst
        self.per_class = per_class
        self.device = device
        self.beta = beta
        self.lr=lr
        self.dis_IB=dis_IB
        self.ch_IB=ch_IB
        self.n_task=n_task
        self.x_list=[]
        self.y_list=[]

        self.hidden1 = nn.Linear(shape_x, shape_z).to(self.device)
        self.hidden2 = nn.Linear(shape_z, shape_z).to(self.device)

        if self.ch_IB!='s':
            self.club = CLUBSample(x_dim=shape_z, y_dim=shape_y, hidden_size=shape_z, device=device)
        else:
            self.club = CLUBSample(x_dim=shape_z, y_dim=shape_y*self.n_task, hidden_size=shape_z, device=device)
            
        self.mine = MINE(x_dim=shape_x, y_dim=shape_z, hidden_size=shape_x, device=device)

        self.club_optimizer = optim.Adam(self.club.parameters(), lr=lr)
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=lr)

        special_layers = torch.nn.ModuleList([self.club, self.mine])
        special_layers_params = list(map(id, special_layers.parameters()))
        base_params = filter(lambda p: id(p) not in special_layers_params, self.parameters())
        self.hidden_optimizer = optim.Adam(base_params, lr=lr)


    
    def forward(self, x, y, task_id, cur_id=0, ch='normal'):

        if self.club.p_mu2.weight.shape[0] < (task_id+1)*self.per_class and self.ch_IB!='s':
            if self.dis_IB:
                pass
            else:
                self.x_list=[]
                self.y_list=[]
                self.club.p_mu2=nn.Linear(self.club.p_mu2.weight.shape[1],(task_id+1)*self.per_class).to(self.device)
                self.club.p_logvar2=nn.Linear(self.club.p_logvar2.weight.shape[1],(task_id+1)*self.per_class).to(self.device)
                self.club.p_mu2.reset_parameters()
                self.club.p_logvar2.reset_parameters()

                self.club_optimizer = optim.Adam(self.club.parameters(), lr=self.lr)

                special_layers = torch.nn.ModuleList([self.club, self.mine])
                special_layers_params = list(map(id, special_layers.parameters()))
                base_params = filter(lambda p: id(p) not in special_layers_params, self.parameters())
                
                self.hidden_optimizer = optim.Adam(base_params, lr=self.lr)

                self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=self.lr)
        
        else:
            pass

        if self.dis_IB:
            y = F.one_hot(y - cur_id*self.per_class, self.per_class)
        else:
            if self.ch_IB != 's':
                y = F.one_hot(y, (task_id+1)*self.per_class)
            else:
                y = F.one_hot(y, (self.n_task)*self.per_class)
            
        

        if len(self.x_list) < 1000 and ch=='normal':
            self.x_list.append(x)
            self.y_list.append(y)

        z = self.hidden1(x)
        z = F.relu(z)
        z = self.hidden2(z)
        
        return z

    def train_net(self, e):

        Obj = 0
        cnt = 0
        if len(self.x_list) > 0:
            for x, y in list(zip(self.x_list, self.y_list)):
                
                z = self.hidden1(x)
                z = F.relu(z)
                z = self.hidden2(z)
                I_zy = self.club.forward(z,y)
                I_xz = self.mine.forward(x,z)
                obj = I_zy - self.beta*I_xz 

                if abs(obj) > 1000:
                    obj = torch.clamp(obj, -1000, 1000)
                    pass
                else:
                    self.hidden_optimizer.zero_grad()
                    obj.backward()
                    self.hidden_optimizer.step()

                if e < 10:
                    z = self.hidden1(x)
                    z = F.relu(z)
                    z = self.hidden2(z)
                    loss_xz = self.mine.learning_loss(x,z)
                    if abs(loss_xz) > 1000:
                        loss_xz=torch.tensor([1])
                        pass
                    else:
                        self.mine_optimizer.zero_grad()
                        loss_xz.backward()
                        self.mine_optimizer.step()

                    z = self.hidden1(x)
                    z = F.relu(z)
                    z = self.hidden2(z)
                    loss_zy = self.club.learning_loss(z,y)
                    if abs(loss_zy) > 1000:
                        loss_zy=torch.tensor([1])
                        pass
                    else:
                        self.club_optimizer.zero_grad()
                        loss_zy.backward()
                        self.club_optimizer.step()

                Obj += obj.item()
                cnt += len(y)
        
        self.x_list=[]
        self.y_list=[]

        if cnt > 0:
            return Obj/cnt
        else: 
            return 0


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size, device):
        super(CLUBSample, self).__init__()
        
        self.device=device
        self.p_mu1=nn.Linear(x_dim, hidden_size).to(self.device)
        self.p_mu2=nn.Linear(hidden_size, y_dim).to(self.device)
        self.p_logvar1=nn.Linear(x_dim, hidden_size).to(self.device)
        self.p_logvar2=nn.Linear(hidden_size, y_dim).to(self.device)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu1(x_samples)
        mu = self.relu(mu)
        mu = self.p_mu2(mu)
        
        logvar = self.p_logvar1(x_samples)
        logvar = self.relu(logvar)
        logvar = self.p_logvar2(logvar)
        logvar = self.tanh(logvar)

        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /(logvar.exp() + 1e-6)-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = - (mu - y_samples)**2 / (logvar.exp() + 1e-6)
        negative = - (mu - y_samples[random_index])**2 / (logvar.exp() + 1e-6)
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, device):
        super(MINE, self).__init__()
        self.device=device
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1)).to(self.device)
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        # T0 = T0.tanh() * 10
        # T1 = T1.tanh() * 10
        # lower_bound = T0.mean() - torch.log(T1.exp().mean() + 1e-6)
        
        T1 = T1.view(T1.shape[0])
        T1 = torch.logsumexp(T1, dim=0) - math.log(T1.shape[0])
        lower_bound = T0.mean() - T1

        # compute the negative loss (maximise loss == minimise -loss)
        # lower_bound = torch.clamp(lower_bound, 0, 10)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)