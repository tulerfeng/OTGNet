import numpy as np
import json
import pandas as pd
import torch
import collections
import time

label2id = {}
id2label = {}
business_label = {}
cnt_label = 0

label2num={}

idx2id = {}
cnt_idx = 0

st_year = 2004
ed_year = 2022
new_label_year_set = {}
new_label_year = {}
selected_new_label_year = {}
selected_new_label_year_name = {}

business_file='./yelp_academic_dataset_business.json'
review_file='./yelp_academic_dataset_review.json'
review_file_year_name = {}
review_file_year = {}
review_tmp_year = {}


per_class=3

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

print(sorted(label2num.items(), key=lambda x: x[1], reverse=True))
print(cnt_label)
print(cnt_idx)

for i in range(st_year, ed_year+1):
    new_label_year_set[i] = {}
    new_label_year[i] = {}
    selected_new_label_year[i] = set()
    selected_new_label_year_name[i] = set()
    review_file_year_name[i] = ('./yelp_review_%d.json'%i)
    review_file_year[i] = open(review_file_year_name[i], 'w')
    review_tmp_year[i] = []

with open(review_file,'r', encoding='utf-8') as f:
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
        y = data_struct[0]
        data_struct_init = time.strptime("2004-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
        cur_sec = time.mktime(data_struct) - time.mktime(data_struct_init)

        
        review_tmp_year[y].append(review)

        if y>= st_year and y<=ed_year:
            if label not in new_label_year_set[y]:
                new_label_year_set[y][label]=set()
                new_label_year[y][label] = 0
            new_label_year_set[y][label].add(bid)
            new_label_year[y][label]=len(new_label_year_set[y][label])

        line = f.readline()

for i in range(st_year, ed_year+1):
    new_label_year[i] = sorted(new_label_year[i].items(), key=lambda x: x[1], reverse=True)

    review_tmp_year[i] = sorted(review_tmp_year[i], key = lambda x: time.mktime(time.strptime(x['date'], "%Y-%m-%d %H:%M:%S")))
    for x in review_tmp_year[i]:
        review_file_year[i].write(json.dumps(x)+"\n")
    review_file_year[i].close()

    for x in new_label_year[i][10:100]:
        flag=0
        for k,v in selected_new_label_year_name.items():
            if x[0] in v:
                flag=1
                break
        if flag==0:
            selected_new_label_year[i].add(x)
            selected_new_label_year_name[i].add(x[0])
        if len(selected_new_label_year[i]) >= per_class:
            print(selected_new_label_year[i])
            break