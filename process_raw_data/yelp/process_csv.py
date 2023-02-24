import numpy as np
import json
import pandas as pd

def list_hash_val(x1,x2):
    cnt=1
    tmp=sorted(list(set(x1)|set(x2)))
    q={}
    for val in tmp:
        if val not in q:
            q[val]=cnt
            cnt+=1
    for i in range(len(x1)):
        x1[i]=q[x1[i]]
        x2[i]=q[x2[i]]
    return x1,x2 

def list_hash_order(x1,x2):
    cnt=0
    tmp=x1
    q={}
    for val in tmp:
        if val not in q:
            q[val]=cnt
            cnt+=1
    for i in range(len(x1)):
        x1[i]=q[x1[i]]
        x2[i]=q[x2[i]]
    return x1,x2 


df=pd.read_csv("./yelp.csv")
src=list(df['u'])
dst=list(df['i'])
label_s=list(df['label_u'])
label_d=list(df['label_i'])
src, dst=list_hash_val(src,dst)
label_s,label_d=list_hash_order(label_s,label_d)

df['u']=src
df['i']=dst
df['label_u']=label_s
df['label_i']=label_d
df.to_csv("./yelp.csv",index=False)
