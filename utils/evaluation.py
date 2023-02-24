import numpy as np
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import precision_score, f1_score


def eval_prediction(model, data, task, eval_task, bs, ch, uml, avg='edge', head='single', per_class=3, retrain=False):
    val_acc, val_ap, val_f1 = [], [], []
    val_acc2, val_ap2, val_f12 = [], [], []
    task=min(5, task)
    if head == 'single':
        n_class = (task+1) * per_class
    elif head == 'multi':
        n_class = per_class 
    node_mp = {}
    node_mp2 ={}
    true_node_mp={}
    true_node_mp2={}
    with torch.no_grad():
        num_batch = math.ceil(len(data.src) / bs)
        for i in range(num_batch):
            st_idx = i * bs
            ed_idx = min((i + 1) * bs, len(data.src))
            if ed_idx==st_idx:
                break 
            src_batch = data.src[st_idx:ed_idx]
            dst_batch = data.dst[st_idx:ed_idx]
            edge_batch = data.edge_idxs[st_idx:ed_idx]
            timestamps_batch = data.timestamps[st_idx:ed_idx]
            
            if ch=='test':
                if uml:
                    src_logits, dst_logits, _,  src_logits2, dst_logits2, _, = model(src_batch, dst_batch, edge_batch, timestamps_batch, task, ch=ch, mask_node=data.induct_nodes, eval_task=eval_task)
                else:    
                    src_logits, dst_logits, _, = model(src_batch, dst_batch, edge_batch, timestamps_batch, task, ch=ch, mask_node=data.induct_nodes, eval_task=eval_task)
            else:
                if uml:
                    src_logits, dst_logits, _, src_logits2, dst_logits2, _, = model(src_batch, dst_batch, edge_batch, timestamps_batch, task, ch='normal', mask_node=None, eval_task=eval_task)
                else:  
                    src_logits, dst_logits, _, = model(src_batch, dst_batch, edge_batch, timestamps_batch, task, ch='normal', mask_node=None, eval_task=eval_task)
            
            
            task_mp_src = torch.tensor([False for i in range(len(src_batch))])
            task_mp_dst = torch.tensor([False for i in range(len(dst_batch))])
            for c in range(eval_task * per_class, (eval_task + 1) * per_class):
                task_mp_src = task_mp_src | (torch.tensor(data.labels_src[st_idx:ed_idx]) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(data.labels_dst[st_idx:ed_idx]) == c)
            if retrain:
                task_mp_src = torch.tensor([True for i in range(len(src_batch))])
                task_mp_dst = torch.tensor([True for i in range(len(dst_batch))])
            src_pred_score = F.softmax(src_logits, dim=1)[task_mp_src]
            dst_pred_score = F.softmax(dst_logits, dim=1)[task_mp_dst]
            src_pred_label = torch.argmax(src_pred_score, dim=1).cpu()
            dst_pred_label = torch.argmax(dst_pred_score, dim=1).cpu()
            src_true_label = torch.tensor(data.labels_src[st_idx:ed_idx]).cpu()[task_mp_src]
            dst_true_label = torch.tensor(data.labels_dst[st_idx:ed_idx]).cpu()[task_mp_dst]
            if head == 'multi' and retrain==False:
                src_true_label = src_true_label - per_class*task
                dst_true_label = dst_true_label - per_class*task
            if retrain:
                n_class = (task+1) * per_class
            src_f1=f1_score(src_true_label, src_pred_label, labels=[i for i in range(0, n_class)], average='micro')
            dst_f1=f1_score(dst_true_label, dst_pred_label, labels=[i for i in range(0, n_class)], average='micro')
            src_ap=precision_score(src_true_label, src_pred_label, labels=[i for i in range(0, n_class)], average='micro')
            dst_ap=precision_score(dst_true_label, dst_pred_label, labels=[i for i in range(0, n_class)], average='micro')
            src_acc=(src_pred_label==src_true_label).float().mean()
            dst_acc=(dst_pred_label==dst_true_label).float().mean()
            if avg=='node':
                for k in range(len(src_true_label)):
                    if src_pred_label[k]==src_true_label[k]:
                        node_mp[src_batch[k]]=1
                    else:
                        node_mp[src_batch[k]]=0
                for k in range(len(dst_true_label)):
                    if dst_pred_label[k]==dst_true_label[k]:
                        node_mp[dst_batch[k]]=1
                    else:
                        node_mp[dst_batch[k]]=0
            f1=(src_f1+dst_f1)/2*100
            ap=(src_ap+dst_ap)/2*100
            acc=(src_acc+dst_acc)/2*100
            val_acc.append(acc)
            val_ap.append(ap)
            val_f1.append(f1)

            if uml:
                src_pred_score2 = F.softmax(src_logits2, dim=1)
                dst_pred_score2 = F.softmax(dst_logits2, dim=1)
                src_pred_label2 = torch.argmax(src_pred_score2, dim=1).cpu()
                dst_pred_label2 = torch.argmax(dst_pred_score2, dim=1).cpu()
                src_true_label2 = torch.tensor(data.labels_src[st_idx:ed_idx]).cpu()[task_mp_src]
                dst_true_label2 = torch.tensor(data.labels_dst[st_idx:ed_idx]).cpu()[task_mp_dst]
                if head == 'multi':
                    src_true_label2 = src_true_label2 - per_class*task
                    dst_true_label2 = dst_true_label2 - per_class*task
                src_f12=f1_score(src_true_label2, src_pred_label2, labels=[i for i in range(0, n_class)], average='micro')
                dst_f12=f1_score(dst_true_label2, dst_pred_label2, labels=[i for i in range(0, n_class)], average='micro')
                src_ap2=precision_score(src_true_label2, src_pred_label2, labels=[i for i in range(0, n_class)], average='micro')
                dst_ap2=precision_score(dst_true_label2, dst_pred_label2, labels=[i for i in range(0, n_class)], average='micro')
                src_acc2=(src_pred_label2==src_true_label2).float().mean()
                dst_acc2=(dst_pred_label2==dst_true_label2).float().mean()
                if avg=='node':
                    for k in range(len(src_true_label2)):
                        if src_pred_label2[k]==src_true_label2[k]:
                            node_mp2[src_batch[k]]=1
                        else:
                            node_mp2[src_batch[k]]=0
                    for k in range(len(dst_true_label2)):
                        if dst_pred_label2[k]==dst_true_label2[k]:
                            node_mp2[dst_batch[k]]=1
                        else:
                            node_mp2[dst_batch[k]]=0
                f12=(src_f12+dst_f12)/2*100
                ap2=(src_ap2+dst_ap2)/2*100
                acc2=(src_acc2+dst_acc2)/2*100
                val_acc2.append(acc2)
                val_ap2.append(ap2)
                val_f12.append(f12)

        
            if ch=='test':
                tmp_src_mask = [x in data.induct_nodes for x in src_batch[task_mp_src]]
                tmp_dst_mask = [x in data.induct_nodes for x in dst_batch[task_mp_dst]]
                # print('out',len(src_batch[tmp_src_mask]))
                # print(src_batch[tmp_src_mask])
                # print('out',len(dst_batch[tmp_dst_mask]))
                # print(dst_batch[tmp_dst_mask])
                if avg=='node':
                    for k in range(len(src_true_label[tmp_src_mask])):
                        if src_pred_label[tmp_src_mask][k]==src_true_label[tmp_src_mask][k]:
                            true_node_mp[src_batch[k]]=1
                        else:
                            true_node_mp[src_batch[k]]=0
                    for k in range(len(dst_true_label[tmp_dst_mask])):
                        if dst_pred_label[tmp_dst_mask][k]==dst_true_label[tmp_dst_mask][k]:
                            true_node_mp[dst_batch[k]]=1
                        else:
                            true_node_mp[dst_batch[k]]=0
                            
                if avg=='node' and uml:
                    for k in range(len(src_true_label2[tmp_src_mask])):
                        if src_pred_label2[tmp_src_mask][k]==src_true_label2[tmp_src_mask][k]:
                            true_node_mp2[src_batch[k]]=1
                        else:
                            true_node_mp2[src_batch[k]]=0
                    for k in range(len(dst_true_label2[tmp_dst_mask])):
                        if dst_pred_label2[tmp_dst_mask][k]==dst_true_label2[tmp_dst_mask][k]:
                            true_node_mp2[dst_batch[k]]=1
                        else:
                            true_node_mp2[dst_batch[k]]=0

                
        if ch=='test':
            true_acc = np.mean(np.array(list(true_node_mp.values()))*100)
            if uml:
                true_acc_m = np.mean(np.array(list(true_node_mp2.values()))*100)
                return true_acc,true_acc,true_acc, np.mean(np.array(list(node_mp2.values()))*100)
            else:
                return true_acc,true_acc,true_acc,0
        
        if avg == 'node':
            node_pred=np.array(list(node_mp.values()))*100
            if uml:
                node_pred2=np.array(list(node_mp2.values()))*100
                return np.mean(node_pred), np.mean(node_pred), np.mean(node_pred), np.mean(node_pred2)
            return np.mean(node_pred), np.mean(node_pred), np.mean(node_pred), 0

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1)
