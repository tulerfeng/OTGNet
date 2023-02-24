from random import gammavariate, random, sample
import torch
import numpy as np
from torch import device, nn, zero_
import torch.nn.functional as F
from modules.memory import Memory
from modules.buffer import Triad_Buffer
from torch.distributions import Bernoulli
from torch.autograd import Variable
from torch.distributions import Normal
from models.IB import IB
from models.pseudo_gen import PGen
import random
from models.TimeEncoder import TimeEncoder


class CIGNN(nn.Module):
  def __init__(self, n_nodes, n_neighbors, batchsize, mem_size, 
               node_init_dim, edge_dim, edge_emb_dim, n_mc, is_r, hidden_dim, node_emb_dim, message_dim,
               per_class, edge_feature, node_feature, node_label, edge_timestamp, label_src, label_dst, node_src, node_dst, n_task, n_interval,
               use_mem, use_feature, use_time, use_IB, dis_IB, pattern_rho, mem_method, class_balance, head, feature_iter, model, radius, beta, gamma, uml,  recover,pmethod,
               dataset, full_data, test_data, ch_IB, select=None, device='cpu'):
    super(CIGNN, self).__init__()
    torch.manual_seed(0)
    # graph
    self.n_nodes = n_nodes
    # Dimensions
    self.node_init_dim = node_init_dim
    self.edge_dim = edge_dim
    self.edge_emb_dim = edge_emb_dim
    self.hidden_dim = hidden_dim
    self.node_emb_dim = node_emb_dim
    self.message_dim = message_dim
    self.dis_IB = dis_IB
    self.ch_IB = ch_IB
    # self.node_emb_dim = node_init_dim
    # self.message_dim = node_init_dim
    # memory
    self.node_feature=torch.tensor(node_feature, dtype=torch.float).to(device)
    self.edge_feature = torch.tensor(edge_feature, dtype=torch.float).to(device)
    self.memory = Memory(n_nodes=self.n_nodes, emb_dim=self.node_emb_dim, device=device, features=None)
    self.triad_buffer = Triad_Buffer(n_edges=len(label_src), n_task=n_task, per_class=per_class, label_src=label_src, label_dst=label_dst, node_src=node_src, node_dst=node_dst, edge_timestamp=edge_timestamp, memory=self.memory.emb, n_vol=mem_size,radius=radius, gamma=gamma)
    if self.dis_IB:
        self.IB = nn.ModuleList([IB(shape_x=self.node_emb_dim, shape_z=self.node_emb_dim, shape_y=per_class, label_src=label_src, label_dst=label_dst, per_class=per_class, device=device, beta=beta, dis_IB=dis_IB, ch_IB=ch_IB, n_task=n_task) for i in range(n_task)])
    else:
        self.IB = IB(shape_x=self.node_emb_dim, shape_z=self.node_emb_dim, shape_y=per_class, label_src=label_src, label_dst=label_dst, per_class=per_class, device=device, beta=beta, dis_IB=dis_IB, ch_IB=ch_IB, n_task=n_task)

    self.device = device
    self.model = model
    self.criterion_list = nn.CrossEntropyLoss(reduction='none').to(self.device)
    self.criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)
    self.bce = nn.BCELoss(reduction='mean').to(self.device)
    self.label_src=torch.tensor(label_src).to(self.device)
    self.label_dst=torch.tensor(label_dst).to(self.device)
    self.node_label=torch.tensor(node_label).to(self.device)
    self.n_mc = n_mc
    self.use_mem = use_mem
    self.use_feature = use_feature
    self.use_time = use_time
    self.n_task=n_task
    self.is_r=is_r
    self.per_class=per_class
    self.n_class=(n_task+1)*per_class
    self.batchsize=batchsize
    self.n_neighbors = n_neighbors
    self.select = select
    self.temperature = 5
    self.eps = 1e-10
    self.confidence=[0,0]
    self.use_IB=use_IB
    self.pattern_rho=pattern_rho
    self.mem_method=mem_method
    self.class_balance=class_balance
    self.head = head
    self.feature_iter = feature_iter
    self.uml=uml
    self.recover=recover
    self.pmethod=pmethod
    self.dataset=dataset
    self.full_data=full_data
    self.test_data=test_data
    if dataset=='yelp':
        self.npr=True
    else:
        self.npr=False
    self.sc=False
    
    if self.use_time == 0:
        self.W_q = nn.Parameter(torch.zeros((self.node_emb_dim, self.node_emb_dim)).to(self.device))
        self.W_k = nn.Parameter(torch.zeros((self.node_emb_dim, self.node_emb_dim)).to(self.device))
    else:
        self.W_q = nn.Parameter(torch.zeros((self.node_emb_dim, self.node_emb_dim)).to(self.device))
        self.W_k = nn.Parameter(torch.zeros((self.node_emb_dim + self.use_time, self.node_emb_dim)).to(self.device))
    self.a = nn.Parameter(torch.zeros(self.node_emb_dim * 2).to(self.device))
    
    self.dropout = nn.Dropout(p=0.5)
    nn.init.xavier_normal_(self.W_q)
    nn.init.xavier_normal_(self.W_k)
    self.W_e = nn.Linear(self.edge_dim, self.edge_emb_dim).to(device)
    if self.npr:
        self.W_uc = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_emb_dim, self.node_emb_dim)).to(self.device))
    else:
        self.W_uc = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_emb_dim*2+self.edge_emb_dim, self.node_emb_dim)).to(self.device)) # update_center
    # self.W_uc2 = nn.Linear(self.node_emb_dim, self.node_emb_dim).to(self.device) 
    nn.init.xavier_normal_(self.W_uc)
    if self.select != 'none':
        self.W_un = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_emb_dim, self.node_emb_dim)).to(self.device)) # update_neighbors
        nn.init.xavier_normal_(self.W_un)
        self.W_p = nn.Parameter(torch.zeros((2 * self.node_emb_dim + self.edge_emb_dim, self.node_emb_dim)).to(self.device)) # propagate
        nn.init.xavier_normal_(self.W_p)
    if self.uml:
        self.pgen = PGen(per_class, self.node_init_dim, self.pmethod, self.node_label, self.node_feature, self.full_data, self.test_data, self.n_task, device)
    # projection
    if self.use_feature == 'fg':
        self.W_c1 = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_init_dim, 128)).to(self.device))
        # self.W_c1 = nn.Parameter(torch.zeros((self.node_emb_dim, 128)).to(self.device))
    elif self.use_feature == 'g':
        self.W_c1 = nn.Parameter(torch.zeros((self.node_emb_dim, 128)).to(self.device))
    elif self.use_feature == 'f':
        self.W_c1 = nn.Parameter(torch.zeros((self.node_init_dim, 256)).to(self.device))
        self.W_c11 = nn.Parameter(torch.zeros((256, 128)).to(self.device))
        nn.init.xavier_normal_(self.W_c11)
    if self.head == 'single':
        self.W_c2 = nn.Parameter(torch.zeros((128, self.per_class))).to(self.device)
    elif self.head == 'multi':
        if not self.sc:
            self.W_c2_list = [nn.Parameter(torch.zeros((128, self.per_class))).to(self.device) for i in range(self.n_task)]
        else:
            self.W_c2_list = [nn.Parameter(torch.zeros((self.node_init_dim + 128, 128))).to(self.device) for i in range(self.n_task)]
            self.W_c3_list = [nn.Parameter(torch.zeros((128, self.per_class))).to(self.device) for i in range(self.n_task)]
        
    if self.use_time > 0:
        self.te = TimeEncoder(expand_dim=use_time, device=self.device) 

    if self.feature_iter:
        self.W_f = nn.Linear(self.node_emb_dim + self.node_init_dim, self.node_emb_dim).to(self.device)

    
    if self.head == 'single':
        nn.init.xavier_normal_(self.W_c1)
        nn.init.xavier_normal_(self.W_c2)
    elif self.head == 'multi':
        nn.init.xavier_normal_(self.W_c1)
        for i in range(n_task):
            nn.init.xavier_normal_(self.W_c2_list[i])
            if self.sc:
                nn.init.xavier_normal_(self.W_c3_list[i])


    # select layers
    self.W_1 = nn.Linear(self.node_emb_dim + self.node_emb_dim * 2 + edge_emb_dim, self.node_emb_dim).to(self.device)
    self.W_2 = nn.Linear(self.node_emb_dim, 1).to(self.device)
    # reinforce
    self.beta = 0.1

  def forward(self, src_idxs, dst_idxs, edge_idxs, timestamps, task_id, ch='normal', mask_node=None, eval_task=0):
    if self.head == 'single':
        tmp_Wc=self.W_c2
        task_id=min(5,task_id)
        if tmp_Wc.shape[1] < (task_id+1)*self.per_class:
            # add_Wc=torch.zeros((tmp_Wc.shape[0],3)).to(self.device)
            # nn.init.xavier_normal_(add_Wc)
            # tmp_Wc=torch.cat((tmp_Wc,add_Wc), dim=1)
            # self.W_c=nn.Parameter(tmp_Wc)
            self.W_c2=nn.Parameter(torch.zeros(tmp_Wc.shape[0],(task_id+1)*self.per_class).to(self.device))
            nn.init.xavier_normal_(self.W_c2)
            
    if self.n_mc>0 :
        src_u, dst_u = self.cal_un(src_idxs, dst_idxs)
        src_u = src_u.detach()
        dst_u = dst_u.detach()
        self.confidence = [1-src_u, 1-dst_u]
    
    if ch=='train':
        eval_task = task_id
    
    if self.model=='OTGNet':
        message = []
        message_b = []
        t_label = []
                
        for i, idxs in enumerate([src_idxs, dst_idxs]): # traverse src and dst in turns
            neighbors, _, n_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_neighbors)
            neighbors = torch.from_numpy(neighbors).long().to(self.device)
            bs = neighbors.shape[0]
            
                
            neighbor_emb = self.memory.emb[neighbors.flatten()]
            if self.feature_iter:
                neighbor_emb = self.W_f(torch.cat((neighbor_emb,self.node_feature[neighbors.flatten()]), dim=1))
            if self.use_IB:
                if mask_node is not None:
                    cur_c = eval_task * self.per_class
                    cur_mask = (self.node_label[idxs]==cur_c) 
                    for i_c in range(1, self.per_class):
                        cur_mask |= (self.node_label[idxs] == cur_c+i_c)
                    temp_node = idxs
                    test_node = list(mask_node & set(torch.tensor(idxs)[cur_mask].numpy().tolist()))
                    p_label = torch.zeros(len(temp_node), dtype=int).to(self.device)
                    p_max = torch.zeros(len(temp_node)).to(self.device)
                    p_max[:] = -1
                    peak = 0
                    cnt_num = 0
                    if self.uml:
                        src_lg2, dst_lg2 = self.pgen(self.node_feature, self.memory.emb.data, src_idxs, dst_idxs, self.label_src[edge_idxs],self.label_dst[edge_idxs], eval_task, self.neighbor_finder, ch='full')
                        if i==0:
                            p_label = torch.argmax(src_lg2, dim=1)
                            prob = torch.softmax(src_lg2, dim=1)
                            prob_mx,_= torch.max(prob, dim=1)
                            sum_val=0
                            sum_cnt=0
                        else:
                            p_label = torch.argmax(dst_lg2, dim=1)

                    cur_label = self.node_label[idxs].clone()
                    tmp_mask = [x in test_node for x in idxs]
                    cur_label[tmp_mask] = p_label[tmp_mask] + cur_c
                    t_label.append(cur_label)

                c_label = self.node_label[idxs]
                if mask_node is not None:
                    c_label = cur_label
                c_label = c_label.view(bs, 1)
                n_label = self.node_label[neighbors]
                n_mp = n_label != c_label
                zero_mp = n_label != -1
                n_mp = n_mp & zero_mp
                if self.use_IB and torch.sum(n_mp) > 0:
                    n_mp = n_mp.flatten()
                    
                    if not self.dis_IB:
                        n_z = self.IB.forward(x = neighbor_emb.data[n_mp], y = n_label.flatten()[n_mp], task_id = task_id)
                        neighbor_emb[n_mp] = n_z
                    else:
                        flat_n_label = n_label.flatten()
                        task_mask_n = {}
                        for c in range(0, (task_id+1) * self.per_class, self.per_class):
                            t = int(c/self.per_class)
                            task_mask_n[t] = (flat_n_label == c)
                            for i_c in range(1, self.per_class):
                                task_mask_n[t] |= (flat_n_label == c+i_c)
                            task_mask_n[t] = task_mask_n[t] & n_mp
                            if torch.sum(task_mask_n[t]) > 0:
                                n_z = self.IB[t].forward(x = neighbor_emb.data[task_mask_n[t]], y = n_label.flatten()[task_mask_n[t]], task_id = task_id, cur_id=t)
                                neighbor_emb[task_mask_n[t]] = n_z
                
                                                 
            neighbor_emb = neighbor_emb.view(bs, self.n_neighbors, self.node_emb_dim)

            if self.use_time > 0:
                n_times = torch.tensor(timestamps).view(len(timestamps),1) - torch.tensor(n_times)
                n_times[n_times < 0] = 0
                n_times = n_times.to(self.device)
                n_times = self.te(n_times)
                neighbor_emb = torch.cat((neighbor_emb, n_times), dim=2)
                neighbor_emb = neighbor_emb.float()

            # from neighbors
            h_n = torch.matmul(neighbor_emb, self.W_k)
            # from centre node
            center_emb = self.memory.emb[idxs]
            if self.feature_iter:
                center_emb = self.W_f(torch.cat((center_emb,self.node_feature[idxs]), dim=1))
            h_c = torch.matmul(center_emb, self.W_q).unsqueeze(dim=1).repeat(1, self.n_neighbors, 1)
            h_in = torch.cat((h_c, h_n), dim=2)
            # drop out
            h_in = self.dropout(h_in)
            att = F.leaky_relu(torch.matmul(h_in, self.a), negative_slope=0.2)
            att = att.softmax(dim=1).unsqueeze(dim=2).repeat(1, 1, self.node_emb_dim)
            h = h_n * att
                
            h = h.sum(dim=1)
            message.append(h)
        
        for i, idxs in enumerate([src_idxs, dst_idxs]):
            mp = self.label_src[edge_idxs] != self.label_dst[edge_idxs]
            if (mask_node is not None) and (self.use_IB):
                mp = t_label[0] != t_label[1]
            if self.use_IB and torch.sum(mp) > 0:
                              
                if not self.dis_IB:
                    if self.dataset=='reddit' or self.dataset=='yelp':
                        z = self.IB.forward(x = (message[i]).data[mp], y = (self.node_label[idxs])[mp], task_id=task_id)
                        (message[i])[mp] = z
                    else:
                        (message[i])[mp] = (message[i])[mp] 
                else:
                    flat_n_label = self.node_label[idxs]
                    task_mask_n = {}
                    for c in range(0, (task_id+1) * self.per_class, self.per_class):
                        t = int(c/self.per_class)
                        task_mask_n[t] = (flat_n_label == c)
                        for i_c in range(1, self.per_class):
                            task_mask_n[t] |= (flat_n_label == c+i_c)
                        task_mask_n[t] = task_mask_n[t] & mp
                        if torch.sum(task_mask_n[t]) > 0:
                            z = self.IB[t].forward(x = (message[i]).data[task_mask_n[t]], y = (self.node_label[idxs])[task_mask_n[t]], task_id=task_id, cur_id=t)
                            (message[i])[task_mask_n[t]] = z
                    
        
        if self.npr:
            to_updated_src = torch.matmul(torch.cat((self.memory.emb[src_idxs], message[0]), dim=1), self.W_uc).tanh()
            to_updated_dst = torch.matmul(torch.cat((self.memory.emb[dst_idxs], message[1]), dim=1), self.W_uc).tanh()
        else:
            h = torch.cat((message[0], message[1]), dim=1).tanh()
            h_e = self.W_e(self.edge_feature[edge_idxs]).tanh()
            h = torch.cat((h, h_e), dim=1)
            
            to_updated_src = torch.matmul(torch.cat((self.memory.emb[src_idxs], h), dim=1), self.W_uc).tanh()
            to_updated_dst = torch.matmul(torch.cat((self.memory.emb[dst_idxs], h), dim=1), self.W_uc).tanh()
        
        
        src_tmp_emb = self.memory.emb[src_idxs]
        dst_tmp_emb = self.memory.emb[dst_idxs]
            
        self.memory.emb[src_idxs] = to_updated_src
        self.memory.emb[dst_idxs] = to_updated_dst

    # prop
    if self.select != 'none':
        for idxs in [src_idxs, dst_idxs]:
            neighbors, _, _ = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_neighbors)
            # unique_neighbors = np.unique(neighbors)
            neighbors = torch.from_numpy(neighbors).long().to(self.device)
            # unique_neighbors = torch.from_numpy(unique_neighbors).long().to(self.device)
            bs = neighbors.shape[0]
            neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_neighbors, self.node_emb_dim)
            # neighbor_emb [bs][n_neighbors][node_emb_dim]

            h1 = torch.matmul(h, self.W_p)
            h1 = h1.unsqueeze(dim=1).repeat(1, self.n_neighbors, 1)
            h2 = h1 * neighbor_emb

            h2 = h2 / (h2.norm(dim=2).view(bs, self.n_neighbors, -1) + self.eps)
            att = torch.softmax(h2.sum(dim=2), dim=1)
            if self.n_mc>0 :
                h1 = h1 * self.confidence[i].unsqueeze(-1).unsqueeze(-1).repeat(1, h1.shape[1], h1.shape[2])
            changed_emb = h1 * att.unsqueeze(dim=2).repeat(1, 1, self.node_emb_dim)
            x = h.unsqueeze(dim=1).repeat(1, self.n_neighbors, 1)
            # only compute gradients of select layers
            x = torch.cat((neighbor_emb, x), dim=2)
            x.detach_()
            x = self.W_1(x)
            x = x.relu()
            x = self.W_2(x)
            probs = x.sigmoid()
            changed_emb = torch.matmul(torch.cat((self.memory.emb[neighbors.flatten()],
                                                    changed_emb.flatten().view(-1, self.node_emb_dim)), dim=1), self.W_un).tanh()
            if self.select == 'all':
                mask = torch.ones((bs, self.n_neighbors)).to(self.device)
            elif self.select == 'random':
                tmp = torch.full((bs, self.n_neighbors), 0.5).to(self.device)
                mask = Bernoulli(tmp).sample()
            elif self.select == 'none':
                # do not propagate
                mask = torch.zeros((bs, self.n_neighbors)).to(self.device)

            mask = mask.float()
            mask = mask.unsqueeze(dim=2).repeat(1, 1, self.node_emb_dim).flatten().view(-1, self.node_emb_dim)
            self.memory.emb[neighbors.flatten()] = mask * changed_emb + (1 - mask) * self.memory.emb[neighbors.flatten()]
    
    # compute loss
    if self.model == 'OTGNet':
        if self.use_feature == 'fg':
            pre_src_logits = torch.matmul(torch.cat((self.memory.emb[src_idxs], self.node_feature[src_idxs]),dim=1), self.W_c1)
            pre_dst_logits = torch.matmul(torch.cat((self.memory.emb[dst_idxs], self.node_feature[dst_idxs]),dim=1), self.W_c1)
        elif self.use_feature == 'g':
            pre_src_logits=torch.matmul(self.memory.emb[src_idxs], self.W_c1)
            pre_dst_logits=torch.matmul(self.memory.emb[dst_idxs], self.W_c1)
        elif self.use_feature == 'f':
            pre_src_logits=torch.matmul(self.node_feature[src_idxs], self.W_c1)
            pre_dst_logits=torch.matmul(self.node_feature[dst_idxs], self.W_c1)

            pre_src_logits=F.relu(pre_src_logits)
            pre_src_logits=torch.matmul(pre_src_logits, self.W_c11)
            pre_dst_logits=F.relu(pre_dst_logits)
            pre_dst_logits=torch.matmul(pre_dst_logits, self.W_c11)

        if self.uml and ch !='inf':

            src_logits2, dst_logits2, loss_s2, loss_d2 = self.pgen(self.node_feature, self.memory.emb.data, src_idxs, dst_idxs, self.label_src[edge_idxs], self.label_dst[edge_idxs], eval_task, self.neighbor_finder)

    pre_src_logits = F.relu(pre_src_logits)
    pre_dst_logits = F.relu(pre_dst_logits)
    pre_src_logits = self.dropout(pre_src_logits)
    pre_dst_logits = self.dropout(pre_dst_logits)

    cur_label_src = self.label_src[edge_idxs]
    cur_label_dst = self.label_dst[edge_idxs]

    if self.head == 'single':
        src_logits = torch.matmul(pre_src_logits, self.W_c2)
        dst_logits = torch.matmul(pre_dst_logits, self.W_c2)
        loss_s = self.criterion_list(src_logits, self.label_src[edge_idxs])
        loss_d = self.criterion_list(dst_logits, self.label_dst[edge_idxs])
    elif self.head == 'multi':
        loss_s = torch.zeros(len(cur_label_src)).to(self.device)
        loss_d = torch.zeros(len(cur_label_dst)).to(self.device)
        src_logits = torch.zeros((len(cur_label_src), self.per_class)).to(self.device)
        dst_logits = torch.zeros((len(cur_label_dst), self.per_class)).to(self.device)
        task_mask_src = {}
        task_mask_dst = {}
        for c in range(0, (task_id+1) * self.per_class, self.per_class):
            t = int(c/self.per_class)
            task_mask_src[t] = (cur_label_src == c)
            task_mask_dst[t] = (cur_label_dst == c)
            for i_c in range(1, self.per_class):
                task_mask_src[t] |= (cur_label_src == c+i_c)
                task_mask_dst[t] |= (cur_label_dst == c+i_c)
                
            if not self.sc:    
                src_logits[task_mask_src[t]] = torch.matmul(pre_src_logits[task_mask_src[t]], self.W_c2_list[t])
                dst_logits[task_mask_dst[t]] = torch.matmul(pre_dst_logits[task_mask_dst[t]], self.W_c2_list[t])
            else:
                                
                src_logits_p = torch.matmul(torch.cat((pre_src_logits[task_mask_src[t]], self.node_feature[src_idxs][task_mask_src[t]]),dim=1), self.W_c2_list[t])
                dst_logits_p = torch.matmul(torch.cat((pre_dst_logits[task_mask_dst[t]], self.node_feature[dst_idxs][task_mask_dst[t]]),dim=1), self.W_c2_list[t])
                src_logits_p = F.relu(src_logits_p)
                dst_logits_p = F.relu(dst_logits_p)
                
                src_logits_p = self.dropout(src_logits_p)
                dst_logits_p = self.dropout(dst_logits_p)
                
                src_logits[task_mask_src[t]] = torch.matmul(src_logits_p, self.W_c3_list[t])
                dst_logits[task_mask_dst[t]] = torch.matmul(dst_logits_p, self.W_c3_list[t])
            
            loss_s[task_mask_src[t]] = self.criterion_list(src_logits[task_mask_src[t]], cur_label_src[task_mask_src[t]] - t*self.per_class)
            loss_d[task_mask_dst[t]] = self.criterion_list(dst_logits[task_mask_dst[t]], cur_label_dst[task_mask_dst[t]] - t*self.per_class)
    
    oldsize=0

    if self.mem_method == 'triad':
        oldsize = torch.sum(torch.tensor(self.triad_buffer.triad_mask))

    if self.class_balance and oldsize > 0 and ch=='train':
        newsize = len(loss_s) - oldsize
        class_mask_src = {}
        class_mask_dst = {}
        class_num_src = {}
        class_num_dst = {}
        for c in range(task_id * self.per_class, (task_id+1) * self.per_class):
            class_mask_src[c] = cur_label_src == c
            class_mask_dst[c] = cur_label_dst == c
            class_num_src[c] = torch.sum(class_mask_src[c])
            class_num_dst[c] = torch.sum(class_mask_dst[c])
            loss_s[class_mask_src[c]] = (newsize - class_num_src[c])/newsize * loss_s[class_mask_src[c]]
            loss_d[class_mask_dst[c]] = (newsize - class_num_dst[c])/newsize * loss_d[class_mask_dst[c]]

        if self.uml:
            class_mask_src = {}
            class_mask_dst = {}
            class_num_src = {}
            class_num_dst = {}

    loss_s = loss_s.unsqueeze(-1)
    loss_d = loss_d.unsqueeze(-1)

    if self.uml and ch!='inf':
        loss_s2 = loss_s2.unsqueeze(-1)
        loss_d2 = loss_d2.unsqueeze(-1)
        loss2 = loss_s2.mean() + loss_d2.mean()

    if task_id > 0 and oldsize > 0 and ch=='train':
        newsize = len(loss_s) - oldsize
        ratio= newsize / (newsize + oldsize)
        ratio = 0.5
        loss = (1-ratio)*loss_s[:newsize].mean() + ratio*loss_s[newsize:].mean()
        loss += (1-ratio)*loss_d[:newsize].mean() + ratio*loss_d[newsize:].mean()
    else:
        loss = loss_s.mean() + loss_d.mean()
   
   
    if self.pattern_rho > 0 and len(self.triad_buffer.pos_src) > 0:
        loss_p = self.get_pattern_loss()
        loss += self.pattern_rho * loss_p
        
    if self.recover and ch == 'test':
        self.memory.emb[src_idxs] = src_tmp_emb
        self.memory.emb[dst_idxs] = dst_tmp_emb

    if ch=='inf':
        return src_logits, dst_logits, loss, loss_s.squeeze(-1), loss_d.squeeze(-1)

    if self.uml:
        return src_logits, dst_logits, loss, src_logits2, dst_logits2, loss2
    
    return src_logits, dst_logits, loss
    

    

  def get_pattern_loss(self):
    pos_score = torch.sum(self.memory.emb[self.triad_buffer.pos_src] * self.memory.emb[self.triad_buffer.pos_dst], dim=1)
    neg_score = torch.sum(self.memory.emb[self.triad_buffer.neg_src] * self.memory.emb[self.triad_buffer.neg_dst], dim=1)
    pos_prob = pos_score.sigmoid()
    neg_prob = neg_score.sigmoid()
    pos_label = torch.ones(len(pos_score), dtype=torch.float, device=self.device)
    neg_label = torch.zeros(len(neg_score), dtype=torch.float, device=self.device)
    loss_p = self.bce(pos_prob, pos_label) + self.bce(neg_prob, neg_label)
    loss_p = loss_p/len(self.triad_buffer.pos_src)
    return loss_p

  def get_reward(self, idxs, neighbor_emb):
    central_emb = self.memory.emb[idxs].repeat(1, self.n_neighbors).view(-1, self.node_emb_dim)
    central_emb_norm = F.normalize(central_emb, p=2, dim=1).detach()
    neighbor_emb_norm = F.normalize(neighbor_emb, p=2, dim=1)
    cos_sim = torch.matmul(central_emb_norm, neighbor_emb_norm.t())
    return cos_sim.mean()

  def compute_sim(self, src_idxs):
    src_norm = F.normalize(self.memory.emb[src_idxs], p=2, dim=1)
    # [bs,node_emb_dim]
    emb_norm = F.normalize(self.memory.emb, p=2, dim=1)
    # [n_nodes,node_emb_dim]

    cos_sim = torch.matmul(src_norm, emb_norm.t())
    sorted_cos_sim, idx = cos_sim.sort(descending=True)
    return sorted_cos_sim, idx

  def compute_score(self, src_idxs, dst_idxs, neg_idxs):
    pos_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[dst_idxs], dim=1)
    neg_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[neg_idxs], dim=1)
    return pos_score.sigmoid(), neg_score.sigmoid()

  def cal_un(self, src_idxs, dst_idxs):
    src_p=torch.zeros(self.W_c2.shape[1]).to(self.device)
    dst_p=torch.zeros(self.W_c2.shape[1]).to(self.device)
    for mc_time in range(self.n_mc):
        if self.use_feature == 'fg':
            pre_src_logits = torch.matmul(torch.cat((self.memory.emb[src_idxs], self.node_feature[src_idxs]),dim=1), self.W_c1)
            pre_dst_logits = torch.matmul(torch.cat((self.memory.emb[dst_idxs], self.node_feature[dst_idxs]),dim=1), self.W_c1)
        elif self.use_feature == 'g':
            pre_src_logits=torch.matmul(self.memory.emb[src_idxs], self.W_c1)
            pre_dst_logits=torch.matmul(self.memory.emb[dst_idxs], self.W_c1)
        elif self.use_feature == 'f':
            pre_src_logits=torch.matmul(self.node_feature[src_idxs], self.W_c1)
            pre_dst_logits=torch.matmul(self.node_feature[dst_idxs], self.W_c1)
        

        pre_src_logits = F.relu(pre_src_logits)
        pre_dst_logits = F.relu(pre_dst_logits)
        pre_src_logits = self.dropout(pre_src_logits)
        pre_dst_logits = self.dropout(pre_dst_logits)
        src_logits = torch.matmul(pre_src_logits, self.W_c2)
        dst_logits = torch.matmul(pre_dst_logits, self.W_c2)
        tmp_src_p = F.softmax(src_logits, dim=1)
        tmp_dst_p = F.softmax(dst_logits, dim=1)
        src_p = src_p + tmp_src_p
        dst_p = dst_p + tmp_dst_p

    src_p/=self.n_mc
    dst_p/=self.n_mc

    src_log_p=torch.log2(src_p)
    dst_log_p=torch.log2(dst_p)
    src_u = -src_p * src_log_p
    dst_u = -dst_p * dst_log_p
    src_u = src_u.sum(dim=1)/torch.log2(torch.tensor(1.0*self.W_c2.shape[1]))
    dst_u = dst_u.sum(dim=1)/torch.log2(torch.tensor(1.0*self.W_c2.shape[1]))
    
    return src_u, dst_u

  def reset_graph(self, nodes=None):    
    self.memory.__init_memory__(nodes)

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder

  def detach_memory(self):
    self.memory.detach_memory()

  def back_up_memory(self):
    return self.memory.emb.clone()

  def restore_memory(self, back_up):
    self.memory.emb = nn.Parameter(back_up)

  def back_up_IB(self):
    return self.IB

  def restore_IB(self, IB):
    self.IB = IB

  def back_up_PGen(self):
    return self.pgen

  def restore_PGen(self, pgen):
    self.pgen = pgen