import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import get_neighbor_finder
import math

class PGen(nn.Module):
    def __init__(self, per_class, node_init_dim, pmethod, node_label, node_feature, full_data, test_data, n_task, device):
        super(PGen, self).__init__()
        self.per_class = per_class
        self.node_init_dim = node_init_dim
        self.device = device
        self.mx_task = 0
        self.lr = 1e-3 # 1e-4
        self.pmethod = pmethod
        self.node_label = node_label
        self.node_feature = node_feature
        self.full_data = full_data
        self.test_data = test_data
        self.n_task = n_task
        self.dropout = nn.Dropout(p=0.5)
        if self.pmethod=='knn' or self.pmethod=='tknn':
            if self.pmethod == 'knn':
                self.n_neighbors = 50
            elif self.pmethod == 'tknn':
                self.n_neighbors = 50
            self.p_label = [0] * len(self.node_label)
            self.p_label[0] = -1
            self.p_label = torch.tensor(self.p_label)
            
            mem_label = [False] * len(self.node_label)
            mem_label = torch.tensor(mem_label)
            
            for c_task in range(self.n_task):
                if self.pmethod == 'knn':
                    test_neighbor_finder = get_neighbor_finder(full_data[c_task], False, mask=test_data[c_task])
                elif self.pmethod == 'tknn':
                    test_neighbor_finder = get_neighbor_finder(full_data[c_task], True, mask=test_data[c_task])
                bs = 300
                num_batch = math.ceil(len(full_data[c_task].src) / bs)
                for c_b in range(num_batch):
                    st_idx = c_b * bs
                    ed_idx = min((c_b + 1) * bs, len(full_data[c_task].src))
                    if ed_idx==st_idx:
                        break 
                    src_batch = full_data[c_task].src[st_idx:ed_idx]
                    dst_batch = full_data[c_task].dst[st_idx:ed_idx]
                    edge_batch = full_data[c_task].edge_idxs[st_idx:ed_idx]
                    timestamps_batch = full_data[c_task].timestamps[st_idx:ed_idx]         
                    for i, idxs in enumerate([src_batch, dst_batch]): 
                        if self.pmethod == 'knn':
                            f_timestamps = [1e15] * len(idxs)
                        elif self.pmethod == 'tknn':
                            f_timestamps = timestamps_batch
                        neighbors, _, n_times = test_neighbor_finder.get_temporal_neighbor(idxs, f_timestamps, self.n_neighbors)
                        neighbors = torch.from_numpy(neighbors).long().to(self.device)
                        bs = neighbors.shape[0]
                        neighbor_label = self.node_label[neighbors.flatten()]
                        neighbor_label = neighbor_label.view(bs, self.n_neighbors)
                        pred = []
                        cur_idx = -1
                        for cur_x in neighbor_label:
                            cur_idx += 1
                            cur_mask = (cur_x == c_task*self.per_class)
                            for cnt_c in range(1, self.per_class):
                                cur_mask = cur_mask | (cur_x == (c_task*self.per_class + cnt_c))
                            tmp_count = torch.bincount(cur_x[cur_mask] - c_task*self.per_class)
                            if len(tmp_count)==0:
                                tmp_count=torch.tensor([0])
                                mem_label[idxs[cur_idx]]=True

                            tmp_label = torch.argmax(tmp_count)
                            pred.append(tmp_label)
                        self.p_label[idxs] = torch.tensor(pred)
        
        elif self.pmethod=='mlp':
            self.W_m1 = nn.Parameter(torch.zeros((self.node_init_dim, 256)).to(self.device))
            self.W_m11 = nn.Parameter(torch.zeros((256, 128)).to(self.device))
            self.W_m2 = nn.Parameter(torch.zeros((128, per_class)).to(self.device))
            nn.init.xavier_normal_(self.W_m1)
            nn.init.xavier_normal_(self.W_m11)
            nn.init.xavier_normal_(self.W_m2)
                     
        elif self.pmethod=='nmlp':
            self.node_emb_dim = 100
            self.n_neighbors = 10
            self.use_feature = 'f'
            if self.use_feature == 'f':
                self.W_m1 = nn.Parameter(torch.zeros((self.node_init_dim + self.node_init_dim, 256)).to(self.device))
            else:
                self.W_m1 = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_init_dim, 256)).to(self.device))
            self.W_m11 = nn.Parameter(torch.zeros((256, 128)).to(self.device))
            self.W_m2 = nn.Parameter(torch.zeros((128, per_class)).to(self.device))
            nn.init.xavier_normal_(self.W_m1)
            nn.init.xavier_normal_(self.W_m11)
            nn.init.xavier_normal_(self.W_m2)
                            
                                            
            
        if self.pmethod=='mlp' or self.pmethod=='nmlp':
        
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            self.criterion_list = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, node_feature, node_emb, src_idxs, dst_idxs, src_label, dst_label, task, neighbor_finder, ch='part'):
        
        src_feature=node_feature[src_idxs]
        dst_feature=node_feature[dst_idxs]
        
        
        if task > self.mx_task:
            self.mx_task = task
            if self.pmethod=='mlp' or self.pmethod=='nmlp':
                nn.init.xavier_normal_(self.W_m1)
                nn.init.xavier_normal_(self.W_m11)
                nn.init.xavier_normal_(self.W_m2)  
                self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            
        if ch=='part':
            src_task_mask = src_label == (task*self.per_class)
            dst_task_mask = dst_label == (task*self.per_class)
            for i in range(task*self.per_class, (task+1)*self.per_class):
                src_task_mask |= (src_label == i)
                dst_task_mask |= (dst_label == i)

            cur_label_src = src_label[src_task_mask]
            cur_label_dst = dst_label[dst_task_mask]
            
        
        
        if self.pmethod=='knn' or self.pmethod=='tknn':
            
            if ch=='part':
                src_idxs=src_idxs[src_task_mask.detach().cpu()]
                dst_idxs=dst_idxs[dst_task_mask.detach().cpu()]
            
                    
            for i, idxs in enumerate([src_idxs, dst_idxs]): # traverse src and dst in turns
                pred = F.one_hot(self.p_label[idxs], self.per_class)
                pred = pred.float().to(self.device)
                if i==0:
                    src_logits2 = pred
                else:
                    dst_logits2 = pred    
              
        
        elif self.pmethod=='mlp':

            if ch=='part':

                src_feature = src_feature[src_task_mask]
                dst_feature = dst_feature[dst_task_mask]

            pre_src_logits2=torch.matmul(src_feature, self.W_m1)
            pre_dst_logits2=torch.matmul(dst_feature, self.W_m1)

            pre_src_logits2=F.relu(pre_src_logits2)
            pre_src_logits2=torch.matmul(pre_src_logits2, self.W_m11)
            pre_dst_logits2=F.relu(pre_dst_logits2)
            pre_dst_logits2=torch.matmul(pre_dst_logits2, self.W_m11)

            pre_src_logits2 = F.relu(pre_src_logits2)
            pre_dst_logits2 = F.relu(pre_dst_logits2)
            pre_src_logits2 = self.dropout(pre_src_logits2)
            pre_dst_logits2 = self.dropout(pre_dst_logits2)

            src_logits2 = torch.matmul(pre_src_logits2, self.W_m2)
            dst_logits2 = torch.matmul(pre_dst_logits2, self.W_m2)
            

                    
        elif self.pmethod=='nmlp':
            
            if ch=='part':
                src_idxs=src_idxs[src_task_mask.detach().cpu()]
                dst_idxs=dst_idxs[dst_task_mask.detach().cpu()]
                    
            message = []
            for i, idxs in enumerate([src_idxs, dst_idxs]): # traverse src and dst in turns
                f_timestamps = [1e15] * len(idxs)
                neighbors, _, n_times = neighbor_finder.get_temporal_neighbor(idxs, f_timestamps, self.n_neighbors)
                neighbors = torch.from_numpy(neighbors).long().to(self.device)
                bs = neighbors.shape[0]
                if self.use_feature=='f':
                    neighbor_emb = node_feature[neighbors.flatten()]
                    neighbor_emb = neighbor_emb.view(bs, self.n_neighbors, self.node_init_dim)
                else:
                    neighbor_emb = node_emb[neighbors.flatten()]
                    neighbor_emb = neighbor_emb.view(bs, self.n_neighbors, self.node_emb_dim)
                
                h = neighbor_emb.mean(dim=1)
                message.append(h)
                
            pre_src_logits = torch.matmul(torch.cat((message[0], node_feature[src_idxs]),dim=1), self.W_m1)
            pre_dst_logits = torch.matmul(torch.cat((message[1], node_feature[dst_idxs]),dim=1), self.W_m1)
            pre_src_logits = F.relu(pre_src_logits)
            pre_dst_logits = F.relu(pre_dst_logits)
            pre_src_logits = self.dropout(pre_src_logits)
            pre_dst_logits = self.dropout(pre_dst_logits)
            src_logits2 = torch.matmul(pre_src_logits, self.W_m11)
            dst_logits2 = torch.matmul(pre_dst_logits, self.W_m11)
            
            src_logits2 = torch.matmul(src_logits2, self.W_m2)
            dst_logits2 = torch.matmul(dst_logits2, self.W_m2)
        

        
        if self.pmethod=='mlp' or self.pmethod=='nmlp':

            if ch=='part':
                loss_s2 = self.criterion_list(src_logits2, cur_label_src - task*self.per_class)
                loss_d2 = self.criterion_list(dst_logits2, cur_label_dst - task*self.per_class)
                return src_logits2, dst_logits2, loss_s2, loss_d2
            else:
                return src_logits2, dst_logits2
            
        else:
            
            if ch=='part':
                return src_logits2, dst_logits2, torch.tensor([0.]), torch.tensor([0.])
            else:
                return src_logits2, dst_logits2
            
        
        

    def train_net(self, loss, e):
        
        if self.pmethod=='mlp' or self.pmethod=='nmlp': 
            if (e+1)%100 == 0:
                self.optimizer.param_groups[0]['lr']/=2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            pass