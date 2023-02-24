import torch
import numpy as np
from torch import exp, nn
import random
from torch.autograd import grad
import math
import torch.nn.functional as F
import time
    

class Triad_Buffer(nn.Module):
    def __init__(self, n_edges, n_task, per_class, label_src, label_dst, node_src, node_dst, edge_timestamp, memory, n_vol=3, n_neighbors=20, radius=0.1, gamma=10):
        super(Triad_Buffer, self).__init__()
        self.n_edges=n_edges
        self.n_task=n_task
        self.per_class=per_class
        self.n_class=(n_task+1)*per_class
        self.n_vol=n_vol
        self.n_neighbors=n_neighbors
        self.label_src=label_src
        self.label_dst=label_dst
        self.node_src=node_src
        self.node_dst=node_dst
        self.radius=radius
        self.gamma=gamma
        self.triad_node_idxs=[]
        self.best_mem=memory.cpu().numpy()
        self.edge_timestamp=edge_timestamp
        self.closed_triads={}
        self.open_triads={}
        self.triad_mask=np.array([False for i in range(self.n_edges)])
        self.pos_src=[]
        self.pos_dst=[]
        self.neg_src=[]
        self.neg_dst=[]
        self.class_cnt=[0 for i in range(self.n_class)]
        self.slots=[[] for i in range(self.n_class)]

    def add(self, class_id, exp_id):
        if len(self.slots[class_id])<self.n_vol:
            self.slots[class_id].append(exp_id)
        else:
            idx=random.randint(0,self.n_vol-1)
            self.sup_mask[self.slots[class_id][idx]]=False
            self.slots[class_id][idx]=exp_id

        self.sup_mask[exp_id]=True

    def update_cnt(self, class_id):
        self.class_cnt[class_id]+=1


    def cal_influence(self, data, model, task, mem, neighbor_finder, cur_n_neighbors, triad_select, dataset, uml, sk, logger, bs=100):

        gt=0
        loss_g = 0
        num_batch = math.ceil(len(data.src) / bs)
        for i in range(num_batch):
            model.detach_memory()
            st_idx = i * bs
            ed_idx = min((i + 1) * bs, len(data.src))
            if ed_idx==st_idx:
                break 
            src_batch = data.src[st_idx:ed_idx]
            dst_batch = data.dst[st_idx:ed_idx]
            edge_batch = data.edge_idxs[st_idx:ed_idx]
            timestamp_batch = data.timestamps[st_idx:ed_idx]
            if uml:
                _, _, performance, _, _, _ = model(src_batch, dst_batch, edge_batch, timestamp_batch, task)
            else:
                _, _, performance = model(src_batch, dst_batch, edge_batch, timestamp_batch, task)
            loss_g = loss_g + performance
        loss_g = loss_g / num_batch
        if uml:
            special_layers = torch.nn.ModuleList([model.IB, model.W_1, model.W_2, model.memory, model.W_e, model.pgen])
        else:
            special_layers = torch.nn.ModuleList([model.IB, model.W_1, model.W_2, model.memory, model.W_e])
        special_layers_params = list(map(id, special_layers.parameters()))
        base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
        params = list(base_params)
        g_grads = grad(loss_g, params, allow_unused=True)
        s_test = self.get_inverse_hvp_lissa(model, data, g_grads, params, task, uml, dataset=dataset)

        # Compute influence
        inf_up_loss = {}
        for i in range(len(data.src)):
            model.detach_memory()
            st_idx = i
            ed_idx = min((i + 1), len(data.src))
            if ed_idx==st_idx:
                break 
            src_batch = data.src[st_idx:ed_idx]
            dst_batch = data.dst[st_idx:ed_idx]
            edge_batch = data.edge_idxs[st_idx:ed_idx]
            timestamp_batch = data.timestamps[st_idx:ed_idx]
            _, _, performance, loss_s, loss_d = model(src_batch, dst_batch, edge_batch, timestamp_batch, task, ch='inf')
            cur_s = src_batch[0]
            cur_d = dst_batch[0]
            train_grads_src = grad(loss_s, params, allow_unused=True, retain_graph=True)
            train_grads_dst = grad(loss_d, params, allow_unused=True)

            if cur_s not in inf_up_loss:
                inf = 0
                for train_grad_p, s_test_p in zip(train_grads_src, s_test):
                    assert train_grad_p.shape == s_test_p.shape
                    inf += -torch.mean(train_grad_p * s_test_p)
                inf_up_loss[cur_s] = inf

            if cur_d not in inf_up_loss:
                inf = 0
                for train_grad_p, s_test_p in zip(train_grads_dst, s_test):
                    assert train_grad_p.shape == s_test_p.shape
                    inf += -torch.mean(train_grad_p * s_test_p)
                inf_up_loss[cur_d] = inf

            if len(inf_up_loss)==data.n_unique_nodes:
                break

        mx = -1e20
        mn = 1e20
        for k, v in inf_up_loss.items():
            mx = max(mx, v)
            mn = min(mn, v)
        for k, v in inf_up_loss.items():
            inf_up_loss[k] = (v - mn)/(mx - mn + 1e-10)

        closed_num=[]
        open_num=[]
        for c in range(task*self.per_class, (task+1)*self.per_class):
            closed_triads, open_triads = self.get_positive_triads(data, task, neighbor_finder, inf_up_loss, c)
            if len(closed_triads) < 5:
                closed_triads, _ = self.get_positive_triads(data, task, neighbor_finder, inf_up_loss, c, md='small')
            if len(open_triads) < 5:
                _, open_triads = self.get_positive_triads(data, task, neighbor_finder, inf_up_loss, c, md='small')
            closed_num.append(len(closed_triads))
            open_num.append(len(open_triads))
            inf_closed = {}
            inf_open = {}
            emb_closed = {}
            emb_open = {}
            for x in closed_triads:
                tmp_set = list(set([self.node_src[x[0]], self.node_src[x[1]], self.node_src[x[2]], self.node_dst[x[0]], self.node_dst[x[1]], self.node_dst[x[2]]]))
                inf_closed[x] = 0
                emb_closed[x] = 0
                for nd in tmp_set:
                    inf_closed[x] += inf_up_loss[nd]/len(tmp_set)
                    emb_closed[x] += mem[nd]/len(tmp_set)
            for x in open_triads:
                tmp_set = list(set([self.node_src[x[0]], self.node_src[x[1]], self.node_dst[x[0]], self.node_dst[x[1]]]))
                inf_open[x] = 0
                emb_open[x] = 0
                for nd in tmp_set:
                    inf_open[x] += inf_up_loss[nd]/len(tmp_set)
                    emb_open[x] += mem[nd]/len(tmp_set)

            for it in emb_closed:
                emb_closed[it]=emb_closed[it].clone().to('cpu')
            
            for it in emb_open:
                emb_open[it]=emb_open[it].clone().to('cpu')

            for it in inf_closed:
                inf_closed[it]=inf_closed[it].clone().to('cpu')
            
            for it in inf_open:
                inf_open[it]=inf_open[it].clone().to('cpu')


            t1=time.time()
            self.closed_triads[c]=self.greedy_approximation(inf_closed, emb_closed, triad_select, mx=sk)
            self.open_triads[c]=self.greedy_approximation(inf_open, emb_open, triad_select, mx=sk)
            gt=gt+(time.time()-t1)

            cur_edge_idxs=set()
            cur_node_idxs=[]
            for k in self.closed_triads[c]:
                self.triad_mask[k[0]] = True
                self.triad_mask[k[1]] = True
                self.pos_src.append(self.node_src[k[2]])
                self.pos_dst.append(self.node_dst[k[2]])
                cur_edge_idxs.add(k[0])
                cur_edge_idxs.add(k[1])

            for k in self.open_triads[c]:
                self.triad_mask[k[0]] = True
                self.triad_mask[k[1]] = True
                self.neg_src.append(self.node_dst[k[1]])
                v1=self.node_src[k[1]]
                if self.node_src[k[0]] != v1:
                    self.neg_dst.append(self.node_src[k[0]])
                else:
                    self.neg_dst.append(self.node_dst[k[0]])
                cur_edge_idxs.add(k[0])
                cur_edge_idxs.add(k[1])

            cur_edge_idxs = list(cur_edge_idxs)
            src_neighbors, _, _ = neighbor_finder.get_temporal_neighbor(self.node_src[cur_edge_idxs], self.edge_timestamp[cur_edge_idxs], cur_n_neighbors)
            dst_neighbors, _, _ = neighbor_finder.get_temporal_neighbor(self.node_dst[cur_edge_idxs], self.edge_timestamp[cur_edge_idxs], cur_n_neighbors)
            cur_node_idxs += self.node_src[cur_edge_idxs].tolist()
            cur_node_idxs += self.node_dst[cur_edge_idxs].tolist()
            for src_n in src_neighbors:
                cur_node_idxs += src_n.tolist()
            for dst_n in dst_neighbors:
                cur_node_idxs += dst_n.tolist()

            cur_node_idxs = list(set(cur_node_idxs))
            cur_node_idxs = [int(x) for x in cur_node_idxs]
            self.best_mem[cur_node_idxs]=mem[cur_node_idxs].detach().cpu().numpy()
            self.triad_node_idxs = list(set(self.triad_node_idxs + cur_node_idxs))            

        logger.debug(str(closed_num)+"\n")
        logger.debug(str(open_num)+"\n")

        return self.closed_triads, self.open_triads, gt

            
    def greedy_approximation(self, init_inf_dict, emb_dict, triad_select, mx=200):
        tmp = sorted(init_inf_dict.items(), key = lambda x: x[1], reverse=True)
        if mx>=10000:
            mx=len(tmp)
        inf_dict = {}
        for k, v in tmp[:mx]:
            inf_dict[k]=v
        cover_set = set()
        selected_set = set()
        num = len(inf_dict)
        cur = 0
        dis_list = []
        for t in range(100):
            x1, x2 = random.sample(list(inf_dict.keys()), 2)
            dis_list.append(torch.dist(emb_dict[x1], emb_dict[x2], p=2))
        dis_list = sorted(dis_list)
        if self.radius==0:
            self.radius = dis_list[20]
        

        for i in range(min(self.n_vol,len(inf_dict))):
            delta = -1e15
            best = ()
            for k in inf_dict:
                if k in selected_set:
                    continue
                tmp_delta = inf_dict[k]
                cnt = 0
                for tmp_k in inf_dict:
                    dis = torch.dist(emb_dict[k], emb_dict[tmp_k], p=2)
                    if dis < self.radius and (tmp_k not in cover_set):
                        cnt += 1
                tmp_delta = tmp_delta + self.gamma*cnt/num
                if tmp_delta > delta:
                    delta = tmp_delta
                    best = k
            if triad_select == 'influence':
                cur += delta
                selected_set.add(best)
                for tmp_k in inf_dict:
                    dis = torch.dist(emb_dict[best], emb_dict[tmp_k], p=2)
                    if dis < self.radius and (tmp_k not in cover_set):
                        cover_set.add(tmp_k)
            else:
                if best != ():
                    selected_set.add(best)

        return selected_set

    
    def get_positive_triads(self, data, task, neighbor_finder, inf_up_loss, c, md='none'):

        closed_triads = set()
        open_triads = set()
        for i in range(len(data.src)):
            st_idx = i
            ed_idx = min((i + 1), len(data.src))
            if ed_idx==st_idx:
                break 
            src_batch = data.src[st_idx:ed_idx]
            dst_batch = data.dst[st_idx:ed_idx]
            edge_batch = data.edge_idxs[st_idx:ed_idx]
            # inf_e = inf_up_loss[src_batch[0]] + inf_up_loss[dst_batch[0]]
            if self.label_src[edge_batch[0]] != c or self.label_dst[edge_batch[0]] != c:
                continue 
            if md=='small':
                closed_triads.add((edge_batch[0],edge_batch[0],edge_batch[0]))
                open_triads.add((edge_batch[0],edge_batch[0]))
                continue
            edge_batch = data.edge_idxs[st_idx:ed_idx]
            timestamp_batch = data.timestamps[st_idx:ed_idx]
            src_neighbors, src_edge_idxs, _ = neighbor_finder.get_temporal_neighbor(src_batch, timestamp_batch, self.n_neighbors)
            dst_neighbors, dst_edge_idxs, _ = neighbor_finder.get_temporal_neighbor(dst_batch, timestamp_batch, self.n_neighbors)
            last_src_neighbors, last_src_edge_idxs, _ = neighbor_finder.get_temporal_neighbor(src_batch, [1e15], self.n_neighbors)
            last_dst_neighbors, last_dst_edge_idxs, _ = neighbor_finder.get_temporal_neighbor(dst_batch, [1e15], self.n_neighbors)
            src_neighbors = src_neighbors[0].astype(np.int32)
            src_edge_idxs = src_edge_idxs[0].astype(np.int32)
            dst_neighbors = dst_neighbors[0].astype(np.int32)
            dst_edge_idxs = dst_edge_idxs[0].astype(np.int32)
            last_src_neighbors = last_src_neighbors[0].astype(np.int32)
            last_src_edge_idxs = last_src_edge_idxs[0].astype(np.int32)
            last_dst_neighbors = last_dst_neighbors[0].astype(np.int32)
            last_dst_edge_idxs = last_dst_edge_idxs[0].astype(np.int32)
            for n1 in range(self.n_neighbors):
                u = src_neighbors[n1]
                idu = src_edge_idxs[n1]
                flag = 0
                #if u==0 or inf_up_loss[u] + inf_e < 0 or self.label_src[idu] != c: 
                if u==0  or self.label_src[idu] != c: 
                    if md == 'small' and u != 0:
                        pass
                    else:
                        continue
                for n2 in range(self.n_neighbors):
                    v = dst_neighbors[n2]
                    if v == 0:
                        continue
                    if u == v:
                        closed_triads.add((src_edge_idxs[n1], dst_edge_idxs[n2], edge_batch[0]))
            for n1 in range(self.n_neighbors):
                u = last_src_neighbors[n1]
                idu = last_src_edge_idxs[n1]
                flag = 0
                #if u==0 or inf_up_loss[u] + inf_e < 0 or self.label_src[idu] != c or u == dst_batch[0]:
                if u==0 or self.label_src[idu] != c or u == dst_batch[0]:
                    if md == 'small' and u != 0 and u != dst_batch[0]:
                        pass
                    else:
                        continue
                for n2 in range(self.n_neighbors):
                    v = last_dst_neighbors[n2]
                    if v == 0:
                        continue
                    if u == v:
                        flag = 1
                        break                            
                if flag == 0:
                    open_triads.add((last_src_edge_idxs[n1], edge_batch[0]))
    
        return closed_triads, open_triads            
    
    
    def get_inverse_hvp_lissa(self, model, data, vs, params, task, uml, dataset=None,
                            bs=100,
                            scale=1000000000,
                            damping=0.0,
                            num_repeats=1):

        if dataset=='taobao':
            num_repeats = 1
        inverse_hvp = None
        num_batch = math.ceil(len(data.src) / bs)
        for rep in range(num_repeats):
            cur_estimate = vs
            for i in range(num_batch):
                model.detach_memory()
                st_idx = i * bs
                ed_idx = min((i + 1) * bs, len(data.src))
                if ed_idx==st_idx:
                    break 
                src_batch = data.src[st_idx:ed_idx]
                dst_batch = data.dst[st_idx:ed_idx]
                edge_batch = data.edge_idxs[st_idx:ed_idx]
                timestamp_batch = data.timestamps[st_idx:ed_idx]
                if uml:
                    _, _, loss, _, _, _ = model(src_batch, dst_batch, edge_batch, timestamp_batch, task)
                else:
                    _, _, loss = model(src_batch, dst_batch, edge_batch, timestamp_batch, task)
                hvp = self.hessian_vector_product(ys=loss, params=params, vs=cur_estimate)
                cur_estimate = [v + (1-damping) * ce - hv / scale for (v, ce, hv) in zip(vs, cur_estimate, hvp)]
            inverse_hvp = [hv1 + hv2 / scale for (hv1, hv2) in zip(inverse_hvp, cur_estimate)] \
            if inverse_hvp is not None \
            else [hv2 / scale for hv2 in cur_estimate]
        inverse_hvp = [item / num_repeats for item in inverse_hvp]
        return inverse_hvp
    

    def hessian_vector_product(self, ys, params, vs):
        grads1 = grad(ys, params, create_graph=True, allow_unused=True)
        grads2 = grad(grads1, params, grad_outputs=vs, allow_unused=True)
        return grads2
