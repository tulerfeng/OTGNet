import numpy as np
import json
import pandas as pd
import torch
import torchtext
import collections

vectors=torchtext.vocab.Vectors(name='./glove/glove.42B.300d.txt')
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


src=[]
dst=[]
ts=[]
src_label=[]
src_label_name=[]
dst_label=[]
dst_label_name=[]

label={}
id_label={}
label_first_time={}
label_data_num=[0 for i in range(10000)]
label_data_set=[set() for i in range(10000)]
selected_label=set([])
selected_new_label_month={}
selected_new_label_month_name={}
new_label_month={}
new_label_month_set={}
new_label_month_edge={}
post={}
post_label_name={}
post_vectors={}
post_comments_num={}
author={}
author_post_last_time={}
num_post=0
num_label=0

limit_cnt={}

year='2009'
st_mon=7
n_month=12
per_class=3
mx_time=2592000/31*7

ob=open('ob'+year+".txt",'w')

for mon in range(st_mon,n_month+1):
    new_label_month[mon]={}
    new_label_month_set[mon]={}
    selected_new_label_month[mon]=set([])
    selected_new_label_month_name[mon]=set([])
    if mon <10:
        month="0"+str(mon)
    else:
        month=str(mon)
    with open("./reddit/2009/RC_2009-"+month,'r') as f:
        line=f.readline()
        while line:
            comment=json.loads(line)
            t=comment['created_utc']
            t=int(t)-1230768002
            sr=comment['subreddit']
            a=comment['author']
            p=comment['link_id']

            if a!='[deleted]':
                if sr not in label:
                    num_label+=1
                    label[sr]=num_label
                    id_label[num_label]=sr
                    label_first_time[num_label]=t
                label_data_set[label[sr]].add(p)
                label_data_num[label[sr]]=len(label_data_set[label[sr]])

                if sr not in new_label_month[mon]:
                    new_label_month_set[mon][sr]=set()
                    new_label_month[mon][sr]=0
                new_label_month_set[mon][sr].add(p)
                new_label_month[mon][sr]=len(new_label_month_set[mon][sr])

            line=f.readline()

        new_label_month[mon]=sorted(new_label_month[mon].items(), key=lambda x: x[1], reverse=True)

        for x in new_label_month[mon][20:100]:
            if x[0] == 'Marijuana' or x[0] == 'nsfw':
                continue
            flag=0
            for k,v in selected_new_label_month_name.items():
                if x[0] in v:
                    flag=1
                    break
            if flag==0:
                selected_new_label_month[mon].add(x)
                selected_new_label_month_name[mon].add(x[0])
            if len(selected_new_label_month[mon])>=per_class:
                print(selected_new_label_month[mon])
                break



print("number of classes:",len(label))
print("node per class(all):",sorted(label_data_num,reverse=True))

for mon in range(st_mon, n_month+1):
    new_label_month_edge[mon]={}
    heter=0
    if mon <10:
        month="0"+str(mon)
    else:
        month=str(mon)
    with open("./reddit/2009/RC_2009-"+month,'r') as f:
        line=f.readline()
        while line:
            comment=json.loads(line)
            t=comment['created_utc']
            t=int(t)-1230768002
            sr=comment['subreddit']
            a=comment['author']
            p=comment['link_id']

            if a!='[deleted]':

                if sr not in selected_new_label_month_name[mon]:
                    line=f.readline()
                    continue
                
                body=split_words(comment['body'])
                vec=get_embedding(body)

                if p not in post:
                    num_post+=1
                    post[p]=num_post
                    post_label_name[num_post]=sr
                    post_comments_num[num_post]=0
                    post_vectors[num_post]=torch.tensor([0.0 for i in range(300)])

                post_comments_num[num_post]+=1
                post_vectors[num_post]+=vec
                

                if a not in author:
                    author[a]=set([])
                    author_post_last_time[a]=({})
                author[a].add(p)
                author_post_last_time[a][post[p]]=t

                for k in author[a].copy():
                    s=post[p]
                    d=post[k]
                    if t- author_post_last_time[a][d]>mx_time:
                        author[a].remove(k)
                        continue
                    if s==d:
                        continue

                    ls=label[post_label_name[s]]
                    ld=label[post_label_name[d]]

                    if ls not in limit_cnt:
                        limit_cnt[ls]=1
                    if ld not in limit_cnt:
                        limit_cnt[ld]=1
                    limit_cnt[ls]+=1
                    limit_cnt[ld]+=1

                    mx = 10000
                    if (limit_cnt[ls] > mx or limit_cnt[ld] > mx) and ls==ld:
                        continue

                    # if ls!=ld:
                    #     continue
                    

                    if ls!=ld:
                        heter+=1
                        
                    if post_label_name[s] not in new_label_month_edge[mon]:
                        new_label_month_edge[mon][post_label_name[s]]=0
                    if post_label_name[d] not in new_label_month_edge[mon]:
                        new_label_month_edge[mon][post_label_name[d]]=0 
                    
                    new_label_month_edge[mon][post_label_name[s]]+=1
                    new_label_month_edge[mon][post_label_name[d]]+=1

                    src.append(s)
                    dst.append(d)
                    ts.append(t)
                    src_label.append(label[post_label_name[s]])
                    src_label_name.append(post_label_name[s])
                    dst_label.append(label[post_label_name[d]])
                    dst_label_name.append(post_label_name[d])
            
            line=f.readline()

        new_label_month_edge[mon]=sorted(new_label_month_edge[mon].items(), key=lambda x: x[1], reverse=True)
        print(new_label_month_edge[mon])
        print('heter edge:', heter)


idx=list(range(len(src)))
df=pd.DataFrame({"u":src,"i":dst,"label_u":src_label,"label_name_u":src_label_name,"label_i":dst_label,"label_name_i":dst_label_name,"ts":ts,"idx":idx})
df.to_csv("./reddit/2009_processed/reddit_2009_half.csv",index=False)


sd=set(src+dst)
sd=sorted(list(sd))
mem=[]
mem.append(torch.tensor([0.0 for i in range(300)]).tolist())
for key in sd:
    tmp=post_vectors[key]/post_comments_num[key]
    mem.append(tmp.tolist())
mem=np.array(mem)
np.save("./reddit/2009_processed/reddit_2009_half_node.npy",mem)

print(mem.shape)
print("number of interactions:",len(src))
print("number of nodes:",len(sd))
        

