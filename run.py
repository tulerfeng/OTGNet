import os
import sys
from pathlib import Path


dataset = sys.argv[1]
#dataset = 'reddit'
#dataset = 'yelp'
#dataset = 'taobao'
model = 'OTGNet' 
n_epoch = 500


select = 'none' 
Path("./result/").mkdir(parents=True, exist_ok=True)
bs = 300
if dataset=='taobao':
    bs = 600

degree=5
if dataset=='reddit':
    lr=1e-4
elif dataset=='yelp':
    lr=5e-3
elif dataset=='taobao':
    lr=1e-3

if dataset == 'yelp':
    n_task=5
else:
    n_task=6
n_class=3
if dataset == 'taobao':
    n_task=3
    n_class=30
    n_epoch=100

n_interval=5
n_mc=0
mem_size=10
use_feature='fg' 
use_memory=1
use_time=5
mem_method = 'triad' 
filename_add = 'test'
filename_add += ("_"+model)
device=0
rp_times=1
is_r=0
blurry=0
online=0
use_IB=1
dis_IB=0
ch_IB = 'm' 
pattern_rho=0.5
class_balance=1
eval_avg='node'
head='multi' 
feature_iter = 1
patience = 100
radius=0
beta=0.3
gamma=20
uml=1
pmethod='knn' 
sk=1000
full_n=0
recover=1



cmd = "python train.py --bs {} --dataset {} --n_degree {} --n_epoch {} --lr {} --select {} --n_task {} --n_class {} --n_interval {} --n_mc {}".format(bs, dataset,degree,
n_epoch, lr, select, n_task, n_class,n_interval, n_mc)
cmd += " --model {}".format(model)
cmd += " --use_memory {}".format(use_memory)
cmd += " --use_feature {}".format(use_feature)
cmd += " --use_time {}".format(use_time)
cmd += " --mem_method {}".format(mem_method)
cmd += " --filename_add {}".format(filename_add)
cmd += " --cuda_device {}".format(device)
cmd += " --mem_size {}".format(mem_size)
cmd += " --rp_times {}".format(rp_times)
cmd += " --is_r {}".format(is_r)
cmd += " --blurry {}".format(blurry)
cmd += " --online {}".format(online)
cmd += " --use_IB {}".format(use_IB)
cmd += " --pattern_rho {}".format(pattern_rho)
cmd += " --class_balance {}".format(class_balance)
cmd += " --eval_avg {}".format(eval_avg)
cmd += " --head {}".format(head)
cmd += " --feature_iter {}".format(feature_iter)
cmd += " --patience {}".format(patience)
cmd += " --radius {}".format(radius)
cmd += " --beta {}".format(beta)
cmd += " --gamma {}".format(gamma)
cmd += " --uml {}".format(uml)
cmd += " --sk {}".format(sk)
cmd += " --full_n {}".format(full_n)
cmd += " --recover {}".format(recover)
cmd += " --pmethod {}".format(pmethod)
cmd += " --dis_IB {}".format(dis_IB)
cmd += " --ch_IB {}".format(ch_IB)
os.system(cmd)

