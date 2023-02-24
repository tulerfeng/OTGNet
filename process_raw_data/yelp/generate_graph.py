import numpy as np
import json
import pandas as pd
import torch
import collections
import torchtext
import time


vectors=torchtext.vocab.Vectors(name='../glove/glove.42B.300d.txt')
counter=collections.Counter(vectors.itos)
vocab=torchtext.vocab.Vocab(counter,vectors=vectors)
vectors=vocab.vectors

def split_words(x):
    for dots in [',','.','!',')',';',':']:
        x=x.replace(dots,'')
    x=x.lower()
    x=x.split(' ')
    return x

def get_embedding(x):
    vec=torch.tensor([0.0 for i in range(300)])
    if len(x)==0:
        return vec
    for i in range(len(x)):
        vec+=vectors[vocab.stoi[x[i]]]
    vec=vec/len(x)
    return vec


label2id = {}
id2label = {}
business_label = {}
cnt_label = 0

label2num={}

idx2id = {}
cnt_idx = 0

src=[]
dst=[]
ts=[]
src_label=[]
src_label_name=[]
dst_label=[]
dst_label_name=[]

new_label_year_set = {}
new_label_year = {}
selected_new_label_year = {}
selected_new_label_year_name = {}

user_post={}
user_post_last_time={}

business_vectors={}
business_comments_num={}

new_label_year_edge={}

business_file='./yelp_academic_dataset_business.json'
review_file_year = {}

st_year = 2015
ed_year = 2019
per_class = 3
mx_time = 3600*24*30

with open(business_file,'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        business = json.loads(line)
        idx = business['business_id']
        label = business['categories']
        if idx is None or label is None:
            line = f.readline()
            continue
        if ',' in label:
            label = label[:label.find(',')]

        if idx not in idx2id:
            cnt_idx += 1
            idx2id[idx] = cnt_idx
            business_label[cnt_idx] = label
        
        if label not in label2id:
            label2id[label] = cnt_label
            id2label[cnt_label] = label
            cnt_label += 1
        
        if label not in label2num:
            label2num[label] = 0
        else:
            label2num[label] += 1

        line = f.readline()

print(sorted(label2num.items(), key=lambda x: x[1], reverse=True)[:100])
print(cnt_label)
print(cnt_idx)

for y in range(st_year, ed_year+1):
    new_label_year_set[y] = {}
    new_label_year[y] = {}
    selected_new_label_year[y] = set()
    selected_new_label_year_name[y] = set()

    with open('./yelp_review_%d.json'%y, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            review = json.loads(line)
            bidx = review['business_id']
            if bidx not in idx2id:
                line = f.readline()
                continue
            bid = idx2id[bidx]
            label = business_label[bid]
            uidx = review['user_id']
            comment = review['text']
            date = review['date']
            data_struct = time.strptime(date,"%Y-%m-%d %H:%M:%S")
            data_struct_init = time.strptime("2004-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
            cur_sec = time.mktime(data_struct) - time.mktime(data_struct_init)

            if label not in new_label_year_set[y]:
                new_label_year_set[y][label]=set()
                new_label_year[y][label] = 0
            new_label_year_set[y][label].add(bid)
            new_label_year[y][label]=len(new_label_year_set[y][label])

            line = f.readline()

    new_label_year[y] = sorted(new_label_year[y].items(), key=lambda x: x[1], reverse=True)

    for x in new_label_year[y][:100]:
        flag=0
        for k,v in selected_new_label_year_name.items():
            if x[0] in v:
                flag=1
                break
        if flag==0:
            selected_new_label_year[y].add(x)
            selected_new_label_year_name[y].add(x[0])
        if len(selected_new_label_year[y]) >= per_class:
            print(selected_new_label_year[y])
            break


print("\n generate graph: \n")

for y in range(st_year, ed_year+1):

    new_label_year_edge[y] = {}
    heter=0
    with open('./yelp_review_%d.json'%y, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            review = json.loads(line)
            bidx = review['business_id']
            if bidx not in idx2id:
                line = f.readline()
                continue
            bid = idx2id[bidx]
            label = business_label[bid]
            uidx = review['user_id']
            comment = review['text']
            date = review['date']
            data_struct = time.strptime(date,"%Y-%m-%d %H:%M:%S")
            data_struct_init = time.strptime("2004-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
            cur_sec = time.mktime(data_struct) - time.mktime(data_struct_init)

            if label not in selected_new_label_year_name[y]:
                line=f.readline()
                continue
            
            body=split_words(comment)
            vec=get_embedding(body)

            if bid not in business_comments_num:
                business_comments_num[bid]=0
                business_vectors[bid]=torch.tensor([0.0 for i in range(300)])

            business_comments_num[bid]+=1
            business_vectors[bid]+=vec
            
            
            if uidx not in user_post:
                user_post[uidx]=set()
                user_post_last_time[uidx]=({})
            user_post[uidx].add(bid)
            user_post_last_time[uidx][bid]=cur_sec

            for k in user_post[uidx].copy():
                s = bid
                d = k
                if cur_sec - user_post_last_time[uidx][d] > mx_time:
                    user_post[uidx].remove(d)
                    continue
                if s==d:
                    continue

                ls=business_label[s]
                ld=business_label[d]
                

                if ls!=ld:
                    heter+=1
                    
                if ls not in new_label_year_edge[y]:
                    new_label_year_edge[y][ls]=0
                if ld not in new_label_year_edge[y]:
                    new_label_year_edge[y][ld]=0 
                
                new_label_year_edge[y][ls]+=1
                new_label_year_edge[y][ld]+=1

                src.append(s)
                dst.append(d)
                ts.append(cur_sec)
                src_label.append(label2id[ls])
                src_label_name.append(ls)
                dst_label.append(label2id[ld])
                dst_label_name.append(ld)
        
            line=f.readline()

    new_label_year_edge[y]=sorted(new_label_year_edge[y].items(), key=lambda x: x[1], reverse=True)
    print(new_label_year_edge[y])
    print('heter edge:', heter)


idx=list(range(len(src)))
df=pd.DataFrame({"u":src,"i":dst,"label_u":src_label,"label_name_u":src_label_name,"label_i":dst_label,"label_name_i":dst_label_name,"ts":ts,"idx":idx})
df.to_csv("./yelp.csv",index=False)

sd=set(src+dst)
sd=sorted(list(sd))
mem=[]
mem.append(torch.tensor([0.0 for i in range(300)]).tolist())
for key in sd:
    tmp=business_vectors[key]/business_comments_num[key]
    mem.append(tmp.tolist())
mem=np.array(mem)
np.save("./yelp_node.npy",mem)

print(mem.shape)
print("number of interactions:",len(src))
print("number of nodes:",len(sd))
        