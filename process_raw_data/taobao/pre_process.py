import numpy as np
import json
import pandas as pd
import torch
import collections
import torchtext
import time
import pandas
import random

item_theme_file='./theme_item_pool.csv'
purchase_file='./user_item_purchase_log.csv'
click_file='./theme_click_log.csv'
item_emb_file='./item_embedding.csv'

item_idx2id={}
theme_idx2id={}
item_theme={}

split_time=3600*24*2
mx_time=3600*24*100

n_task=3
per_class=30

f = pandas.read_csv(item_theme_file)
item_idx = list(f['item_id'])
theme_idx = list(f['theme_id'])
cnt1 = 0
cnt2 = 0
for i in range(len(item_idx)):
    if item_idx[i] not in item_idx2id:
        cnt1 += 1
        item_idx2id[item_idx[i]] = cnt1
    if theme_idx[i] not in theme_idx2id:
        theme_idx2id[theme_idx[i]] = cnt2
        cnt2 += 1
    item_theme[item_idx2id[item_idx[i]]] = theme_idx2id[theme_idx[i]]

f = pandas.read_csv(click_file)
new_label_day={}
new_label_day_set={}
selected_new_label_day = {}
selected_new_label_day_name = {}
for i in range(0, n_task):
    new_label_day[i]={}
    new_label_day_set[i]={}
    selected_new_label_day[i] = set()
    selected_new_label_day_name[i] = set()

uidx = list(f['user_id'])
item_idx = list(f['item_id'])
raw_time = list(f['reach_time'])
main_class = list(f['cate_level1_id'])
leaf_class = list(f['leaf_cate_id'])
item_label = {}
label2id = {}
cnt_label = 0
info = []

for i in range(len(uidx)):
    data_struct = time.strptime(str(raw_time[i]), "%Y%m%d%H%M%S")
    data_struct_init = time.strptime("20181207000000","%Y%m%d%H%M%S")
    cur_sec = time.mktime(data_struct) - time.mktime(data_struct_init)
    cur_label = main_class[i]
    cur_label = item_theme[item_idx2id[item_idx[i]]]
    cur_label = leaf_class[i]
    info.append((uidx[i], item_idx2id[item_idx[i]], raw_time[i], cur_sec, cur_label))
    if cur_label not in label2id:
        label2id[cur_label]=cnt_label
        cnt_label+=1
    item_label[item_idx2id[item_idx[i]]]=label2id[cur_label]


info = sorted(info, key=lambda x: x[3])

for i in range(len(info)):
    w = min(int(info[i][3]//split_time), n_task)
    label = label2id[info[i][4]]
    item_id = info[i][1]
    if label not in new_label_day_set[w]:
        new_label_day_set[w][label]=set()
        new_label_day[w][label] = 0
    new_label_day_set[w][label].add(item_id)
    new_label_day[w][label]=len(new_label_day_set[w][label])

task_list=[2,0,1]

for i in task_list:
    new_label_day[i]=sorted(new_label_day[i].items(), key=lambda x: x[1], reverse=True)
    for x in new_label_day[i][:100]:
        flag=0
        for k,v in selected_new_label_day_name.items():
            if x[0] in v:
                flag=1
                break
        if flag==0:
            selected_new_label_day[i].add(x)
            selected_new_label_day_name[i].add(x[0])
        if len(selected_new_label_day[i]) >= per_class:
            print(selected_new_label_day[i])
            print(len(list(selected_new_label_day[i])))
            print(' ')
            break

new_label_day_edge={}
heter={}
user_post={}
user_post_last_time={}

src=[]
dst=[]
ts=[]
src_label=[]
src_label_name=[]
dst_label=[]
dst_label_name=[]

print('generating graph......')

for i in range(len(info)):
    w = min(int(info[i][3]//split_time), n_task)
    item_id = info[i][1]
    uidx = info[i][0]
    cur_time = info[i][2]
    label = item_label[item_id]
    if w not in new_label_day_edge:
        new_label_day_edge[w]={}
        heter[w]=0

    if label not in selected_new_label_day_name[w]:
        continue

    if uidx not in user_post:
        user_post[uidx]=set()
        user_post_last_time[uidx]=({})
    user_post[uidx].add(item_id)
    user_post_last_time[uidx][item_id]=cur_time

    for k in user_post[uidx].copy():
        s = item_id
        d = k
        if cur_time - user_post_last_time[uidx][d] > mx_time:
            user_post[uidx].remove(d)
            continue
        if s==d:
            continue

        ls=item_label[s]
        ld=item_label[d]

        if ls!=ld:
            heter[w]+=1
            
        if ls not in new_label_day_edge[w]:
            new_label_day_edge[w][ls]=0
        if ld not in new_label_day_edge[w]:
            new_label_day_edge[w][ld]=0 
        
        new_label_day_edge[w][ls]+=1
        new_label_day_edge[w][ld]+=1

        src.append(s)
        dst.append(d)
        ts.append(cur_time)
        src_label.append(ls)
        src_label_name.append(ls)
        dst_label.append(ld)
        dst_label_name.append(ld)


for i in range(0, n_task):
    new_label_day_edge[i]=sorted(new_label_day_edge[i].items(), key=lambda x: x[1], reverse=True)
    print(new_label_day_edge[i])
    print('heter edge:', heter[i])

idx=list(range(len(src)))
df=pd.DataFrame({"u":src,"i":dst,"label_u":src_label,"label_name_u":src_label_name,"label_i":dst_label,"label_name_i":dst_label_name,"ts":ts,"idx":idx})
df.to_csv("./taobao.csv",index=False)


f=pd.read_csv(item_emb_file)
item_idx = list(f['item_id'])
emb=list(f['emb'])
item_emb={}
for i in range(len(item_idx)):
    if item_idx[i] in item_idx2id:
        emb[i]=[float(x) for x in emb[i].split(' ')]
        item_emb[item_idx2id[item_idx[i]]]=emb[i]


sd=set(src+dst)
sd=sorted(list(sd))
mem=[]
mem.append(torch.tensor([0.0 for i in range(128)]).tolist())
class_cnt={}
class_mean={}
class_set={}
oov=[]
id=0
for key in sd:
    id += 1
    if key in item_emb:
        tmp=torch.tensor(item_emb[key])
        label=item_label[key]
        if label not in class_cnt:
            class_cnt[label] = 0
            class_mean[label] = torch.tensor([0.0 for i in range(128)])
            class_set[label] = set()
        class_cnt[label] += 1
        class_mean[label] += torch.tensor(item_emb[key])
        class_set[label].add(torch.tensor(item_emb[key]))
    else:
        tmp=torch.tensor([0.0 for i in range(128)])
        oov.append((key,id))
    mem.append(tmp.tolist())

print('OOV:',len(oov))

mem=np.array(mem)
print(mem.shape)
np.save("./taobao_node.npy",mem)

print("number of interactions:",len(src))
print("number of nodes:",len(sd))