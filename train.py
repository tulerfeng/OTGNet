from random import sample
import random
import torch
import numpy as np
import math
from torch.nn import init
from tqdm import tqdm
import time
import pickle
import argparse
from pathlib import Path
from models.OTGNet import CIGNN
from utils.data_processing import get_data, computer_time_statics
from utils.utils import get_neighbor_finder, RandEdgeSampler, EarlyStopMonitor
from utils.evaluation import eval_prediction
from utils.log_and_checkpoints import set_logger, get_checkpoint_path
import matplotlib.pyplot as plt
import seaborn

model = 'OTGNet'
parser = argparse.ArgumentParser('OTGNet')
parser.add_argument('--dataset', type=str,default='reddit')
parser.add_argument('--model', type=str, default='OTGNet', help='Model')
parser.add_argument('--bs', type=int, default=300, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=5, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--select', type=str, default='reinforce', help='Policy select')
parser.add_argument('--n_task', type=int, default=6, help='Number of tasks')
parser.add_argument('--n_class', type=int, default=3, help='Classes per task')
parser.add_argument('--n_interval', type=int, default=3, help='Interval of RL training')
parser.add_argument('--n_mc', type=int, default=3, help='Number of MC Dropout')
parser.add_argument('--use_memory', type=int, default=1, help='Use memory buffer or not')
parser.add_argument('--use_feature', type=str, default='fg', help='Use node feature or not')
parser.add_argument('--use_time', type=int, default=5, help='Use time or not')
parser.add_argument('--mem_method', type=str, default='triad', help='Memory buffer sample method')
parser.add_argument('--filename_add', type=str, default='', help='Attachment to filename')
parser.add_argument('--cuda_device', type=int, default=0, help='Device of cuda')
parser.add_argument('--mem_size', type=int, default=10, help='Size of memory slots')
parser.add_argument('--rp_times', type=int, default=1, help='repeat running times')
parser.add_argument('--is_r', type=int, default=1, help='is_r')
parser.add_argument('--blurry', type=int, default=1, help='blurry setting')
parser.add_argument('--online', type=int, default=1, help='online setting')
parser.add_argument('--use_IB', type=int, default=1, help='use IB')
parser.add_argument('--dis_IB', type=int, default=1, help='dis IB')
parser.add_argument('--ch_IB', type=str, default='m', help='ch IB')
parser.add_argument('--pattern_rho', type=float, default=0.1, help='pattern_rho')
parser.add_argument('--class_balance', type=int, default=1, help='class balance')
parser.add_argument('--eval_avg', type=str, default='node', help='evaluation average')
parser.add_argument('--head', type=str, default='single', help='projection head')
parser.add_argument('--feature_iter', type=int, default=1, help='feature_iter')
parser.add_argument('--patience', type=int, default=100, help='patience')
parser.add_argument('--radius', type=float, default=0, help='radius')
parser.add_argument('--beta', type=float, default=0, help='beta')
parser.add_argument('--gamma', type=float, default=0, help='gamma')
parser.add_argument('--uml', type=int, default=0, help='uml')
parser.add_argument('--pmethod', type=str, default='knn', help='pseudo-label method')
parser.add_argument('--sk', type=int, default=1000, help='number of triads candidates')
parser.add_argument('--full_n', type=int, default=1, help='full_n')
parser.add_argument('--recover', type=int, default=1, help='recover')

log_to_file = True
args = parser.parse_args()
dataset = args.dataset
model = args.model
select = args.select
Epoch = args.n_epoch
Batchsize = args.bs
n_neighbors = args.n_degree
lr = args.lr
n_task = args.n_task
n_class = args.n_class
n_interval = args.n_interval
n_mc = args.n_mc
use_mem = args.use_memory==1
use_feature = args.use_feature
use_time = args.use_time
blurry = args.blurry==1
online = args.online==1
is_r = args.is_r==1
mem_method = args.mem_method
mem_size = args.mem_size
rp_times = args.rp_times
use_IB = args.use_IB==1
dis_IB = args.dis_IB==1
ch_IB = args.ch_IB
pattern_rho = args.pattern_rho
class_balance = args.class_balance
eval_avg = args.eval_avg
head=args.head
feature_iter=args.feature_iter==1
patience=args.patience
radius = args.radius
beta = args.beta
gamma = args.gamma
uml = args.uml==1
pmethod = args.pmethod
sk = args.sk
full_n = args.full_n==1
recover = args.recover==1

avg_performance_all=[]
avg_forgetting_all=[]
task_acc_all=[0 for i in range(n_task)]
task_acc_vary=[[0]*n_task for i in range(n_task)]
task_acc_vary_cur=[[0]*n_task for i in range(n_task)]


for rp in range(rp_times):
    start_time=time.time()
    logger, time_now = set_logger(model, dataset, select, log_to_file)
    Path("log/{}/{}/checkpoints".format(model, time_now)).mkdir(parents=True, exist_ok=True)
    Img_path = "log/{}/{}/checkpoints/result.png".format(model, time_now)
    Loss_path1 = "log/{}/{}/checkpoints/loss1.png".format(model, time_now)
    Loss_path2 = "log/{}/{}/checkpoints/loss2.png".format(model, time_now)
    loss_mem1 = []
    loss_mem2 = []
    f = open("./result/{}.txt".format(dataset+args.filename_add),"a+")
    f.write(str(args))
    f.write("\n")
    f.write(time_now)
    f.write("\n")

    print(str(args))
    # data processing
    node_features, edge_features, full_data, train_data, val_data, test_data, all_data, re_train_data, re_val_data = get_data(dataset,n_task,n_class,blurry)
    sd=int(time.time())%100 # Note that this seed can't fix the results
    np.random.seed(sd)  # cpu vars
    torch.manual_seed(sd)  # cpu  vars
    random.seed(sd)
    torch.cuda.manual_seed(sd)
    torch.backends.cudnn.deterministic=True
    torch.cuda.manual_seed_all(sd)  # gpu vars
    f.write("seed: %d\n"%(sd))
    label_src = all_data.labels_src
    label_dst = all_data.labels_dst
    node_src = all_data.src
    node_dst = all_data.dst
    edge_timestamp = all_data.timestamps
    node_label = [-1 for i in range(all_data.n_unique_nodes + 1)]
    for i in range(len(label_src)):
        node_label[all_data.src[i]]=label_src[i]
        node_label[all_data.dst[i]]=label_dst[i]
    torch.cuda.set_device(args.cuda_device)
    device = 'cuda'
    logger.debug(str(args))
    g_time=0
    sgnn = CIGNN(node_features.shape[0], n_neighbors=n_neighbors, batchsize=Batchsize, mem_size=mem_size,
                    node_init_dim=node_features.shape[1], edge_dim=edge_features.shape[1], edge_emb_dim=1, n_mc=n_mc, is_r=is_r,
                    hidden_dim=100, node_emb_dim=100, message_dim=50, per_class=n_class, 
                    edge_feature=edge_features, node_feature=node_features, node_label=node_label, edge_timestamp=edge_timestamp,
                    label_src=label_src, label_dst=label_dst, node_src=node_src, node_dst=node_dst, n_task=n_task, n_interval=n_interval, 
                    use_mem=use_mem, use_feature=use_feature, use_time=use_time, use_IB=use_IB, dis_IB=dis_IB, pattern_rho=pattern_rho, mem_method=mem_method, class_balance=class_balance,
                    head=head, feature_iter=feature_iter, model=model, radius=radius, beta=beta, gamma=gamma,uml=uml, recover=recover, pmethod=pmethod,
                    dataset=dataset, full_data=full_data, test_data=test_data, ch_IB=ch_IB, select=select, device=device)
    sgnn.to(device)
    

    sgnn.reset_graph()

    logger.debug("./result/{}.txt".format(dataset+args.filename_add))
    LOSS = []
    Policy_LOSS = []
    val_acc, val_ap, val_f1 = [], [], []
    early_stopper = [EarlyStopMonitor(max_round=patience) for i in range(n_task+1)]
    test_best=[0 for i in range(n_task)]
    test_neighbor_finder=[]

    for task in range(0,n_task):
        # initialize temporal graph
        train_neighbor_finder = get_neighbor_finder(train_data[task], False)
        test_neighbor_finder.append(get_neighbor_finder(all_data, False, mask=test_data[task]))
        full_neighbor_finder = get_neighbor_finder(all_data, False)

        if model == 'OTGNet':
            special_layers = torch.nn.ModuleList([sgnn.IB, sgnn.W_1, sgnn.W_2, sgnn.memory])
            special_layers_params = list(map(id, special_layers.parameters()))
            base_params = filter(lambda p: id(p) not in special_layers_params, sgnn.parameters())
            optimizer = torch.optim.Adam(base_params, lr=lr)
        else:
            optimizer = torch.optim.Adam(sgnn.parameters(), lr=lr)

        for e in range(Epoch):
            print("task:",task,"epoch:",e)
            logger.debug('task {} , start {} epoch'.format(task,e))
            num_batch = math.ceil(len(train_data[task].src) / Batchsize)
            Loss = 0
            Obj = 0
            Policy_Loss = 0
            Reward = 0
            #sgnn.reset_graph(full_data[task].unique_nodes)
            sgnn.reset_graph()
            sgnn.set_neighbor_finder(train_neighbor_finder)
            sgnn.train()

            if (e+1)%100 == 0:
                optimizer.param_groups[0]['lr']/=2

            for i in range(num_batch):
                if mem_method == 'triad':
                    sgnn.memory.emb[sgnn.triad_buffer.triad_node_idxs]=torch.tensor(sgnn.triad_buffer.best_mem[sgnn.triad_buffer.triad_node_idxs]).to(device)
                sgnn.detach_memory()
                st_idx = i * Batchsize
                ed_idx = min((i + 1) * Batchsize, len(train_data[task].src))

                if use_mem and mem_method == 'triad' and task > 0:
                    src_batch = np.concatenate((train_data[task].src[st_idx:ed_idx], all_data.src[sgnn.triad_buffer.triad_mask])) 
                    dst_batch = np.concatenate((train_data[task].dst[st_idx:ed_idx], all_data.dst[sgnn.triad_buffer.triad_mask])) 
                    edge_batch = np.concatenate((train_data[task].edge_idxs[st_idx:ed_idx], all_data.edge_idxs[sgnn.triad_buffer.triad_mask]))
                    timestamp_batch = np.concatenate((train_data[task].timestamps[st_idx:ed_idx], all_data.timestamps[sgnn.triad_buffer.triad_mask])) 
                else:
                    src_batch = train_data[task].src[st_idx:ed_idx]
                    dst_batch = train_data[task].dst[st_idx:ed_idx]
                    edge_batch = train_data[task].edge_idxs[st_idx:ed_idx]
                    timestamp_batch = train_data[task].timestamps[st_idx:ed_idx]

                # sgnn.set_neighbor_finder(full_neighbor_finder)

                if uml:
                    _, _, ce_loss, _, _, ce_loss2 = sgnn(src_batch, dst_batch, edge_batch, timestamp_batch, task, ch='train')
                else:
                    _, _, ce_loss = sgnn(src_batch, dst_batch, edge_batch, timestamp_batch, task, ch='train')

                Loss += ce_loss.item()

                loss = ce_loss
                if uml:
                    sgnn.pgen.train_net(ce_loss2, e)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if use_IB:
                    if dis_IB:
                        for dis_i in range(task+1):
                            Obj += sgnn.IB[dis_i].train_net(e)
                    else:
                        Obj += sgnn.IB.train_net(e)

            Loss=Loss / num_batch
            Obj=Obj / num_batch
            loss_mem1.append(Loss)
            loss_mem2.append(Obj)
            print("train loss: %.4f"%(Loss))
            print("obj: %.4f"%(Obj))
            LOSS.append(Loss)
            logger.debug("loss in whole dataset = {}".format(Loss))


            # validation
            sgnn.eval()

            # sgnn.reset_graph(full_data[task].unique_nodes)
            sgnn.reset_graph()
            train_n_acc, train_n_ap, train_n_f1, train_m_acc = eval_prediction(sgnn, train_data[task], task, task, Batchsize, 'train', uml, eval_avg, head, n_class)
            if full_n:
                sgnn.set_neighbor_finder(full_neighbor_finder)
            else:
                sgnn.set_neighbor_finder(test_neighbor_finder[task])
            val_n_acc, val_n_ap, val_n_f1, val_m_acc = eval_prediction(sgnn, val_data[task], task, task, Batchsize, 'val', uml, eval_avg, head, n_class)
            train_memory_backup = sgnn.back_up_memory()
            if model=='OTGNet':
                train_IB_backup = sgnn.back_up_IB()
                if uml:
                    train_PGen_backup = sgnn.back_up_PGen()
                else:
                    train_PGen_backup =None
            else:
                train_IB_backup = None
                train_PGen_backup =None
            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[task], task, task, Batchsize, 'test', uml, eval_avg, head, n_class)
            print("train_acc:%.2f   val_acc:%.2f   test_acc:%.2f"%(train_n_acc,val_n_acc,test_n_acc))
            logger.debug("train_acc:%.2f   val_acc:%.2f   test_acc:%.2f"%(train_n_acc,val_n_acc,test_n_acc))
            val_acc.append(val_n_acc)
            val_ap.append(val_n_ap)
            val_f1.append(val_n_f1)

            sgnn.restore_memory(train_memory_backup)

            if online:
                for k in range(task+1):
                    if head == 'single':
                        test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], task, k, Batchsize, 'test', uml, eval_avg, head, n_class)
                    else:
                        test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], k, k, Batchsize, 'test', uml, eval_avg, head, n_class)
                    test_best[k]=max(test_best[k], test_n_acc)
                    task_acc_vary[k][task]+=test_n_acc
                break
            else:
                if early_stopper[task].early_stop_check(val_n_ap, sgnn, model, train_memory_backup, time_now, task, train_IB_backup, train_PGen_backup) or e == Epoch - 1:
                    logger.info('No improvement over {} epochs, stop training'.format(early_stopper[task].max_round))
                    logger.info(f'Loading the best model at epoch {early_stopper[task].best_epoch}')
                    best_model_path, _, _, _ = get_checkpoint_path(model, time_now, task, uml)
                    sgnn = torch.load(best_model_path)
                    logger.info(f'Loaded the best model at epoch {early_stopper[task].best_epoch} for inference')
                    sgnn.eval()
                    for k in range(task+1):
                        if full_n:
                            sgnn.set_neighbor_finder(full_neighbor_finder)
                        else:
                            sgnn.set_neighbor_finder(test_neighbor_finder[k])
                        best_model_path, best_mem_path, best_IB_path, best_PGen_path = get_checkpoint_path(model, time_now, k, uml)
                        best_mem = torch.load(best_mem_path)
                        sgnn.restore_memory(best_mem)
                        if model=='OTGNet':
                            if not dis_IB:
                                best_IB = torch.load(best_IB_path)
                                if dataset != 'reddit' and dataset != 'yelp':
                                    sgnn.restore_IB(best_IB)
                            if uml:
                                best_PGen = torch.load(best_PGen_path)
                                sgnn.restore_PGen(best_PGen)
                        if head == 'single':
                            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], task, k, Batchsize, 'test', uml, eval_avg, head, n_class)
                        elif head == 'multi':
                            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], k, k, Batchsize, 'test', uml, eval_avg, head, n_class)
                        test_best[k]=max(test_best[k], test_n_acc)
                        task_acc_vary[k][task]+=test_n_acc
                        task_acc_vary_cur[k][task]=test_n_acc
                        print("task %d: "%(k)+str(task_acc_vary_cur[k][task]))
                    break   

        
        if mem_method=='triad' and task != n_task-1:
            c_triad, o_triad, gt=sgnn.triad_buffer.cal_influence(train_data[task], sgnn, task, sgnn.memory.emb, train_neighbor_finder, n_neighbors, 'influence', dataset, uml, sk, logger)
            g_time += gt

    # test
    print('best performance: ',test_best)
    sgnn.eval()
    sgnn.set_neighbor_finder(full_neighbor_finder)
    avg_performance=[]
    avg_forgetting=[]
    for i in range(n_task):
        print("task:", i)
        test_acc, test_ap, test_f1 = task_acc_vary_cur[i][n_task-1], task_acc_vary_cur[i][n_task-1], task_acc_vary_cur[i][n_task-1]
        avg_performance.append(test_acc)
        avg_forgetting.append(test_best[i]-test_acc)
        task_acc_all[i]+=test_acc
        logger.debug("in test, acc = {}".format(test_acc))
        print("in test, acc = {}".format(test_acc))
        f.write("task %d, test_acc=%.2f, test_ap = %.2f, test_f1=%.2f"%(i, test_acc, test_ap, test_f1))
        f.write("\n")
    print('avg performance: ',avg_performance)
    print("Average performance: %.2f"%(np.array(avg_performance).mean()))
    print("Average forgetting: %.2f"%(np.array(avg_forgetting[:-1]).mean()))
    avg_performance_all.append(np.array(avg_performance).mean())
    avg_forgetting_all.append(np.array(avg_forgetting[:-1]).mean())
    f.write("Average performance: %.2f"%(np.array(avg_performance).mean()))
    f.write("\n")
    f.write("Average forgetting: %.2f"%(np.array(avg_forgetting[:-1]).mean()))
    f.write("\n")
    if mem_method=='triad':
        print("greedy_time: ", g_time/3600)
        f.write("greedy_time: "+str(g_time/3600))
        logger.debug(str(c_triad))
        logger.debug(str(o_triad))
    all_time=time.time()-start_time
    print("all_time: ", all_time/3600)
    f.write("all_time: "+str(all_time/3600))

    f.write('train loss:'+str(loss_mem1))
    f.write("\n")
    f.write('IB loss:'+str(loss_mem2))
    f.write("\n")
    plt.plot(list(range(0,len(loss_mem1))), loss_mem1, c='b')
    plt.savefig(Loss_path1)
    plt.show()
    plt.clf()
    plt.plot(list(range(0,len(loss_mem2))), loss_mem2, c='b')
    plt.savefig(Loss_path2)
    plt.show()
    plt.clf()

f.write(str(args))
f.write("\n")
f.write(time_now)
f.write("\n")
print("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
print("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_performance_all).std()))
f.write("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
f.write("\n")
f.write("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_forgetting_all).std()))
f.write("\n")
for i in range(n_task):
    print("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
    f.write("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
    f.write("\n")

c_list=['tomato','golden','pea','leaf','jade','bluish','violet','strawberry']
for i in range(n_task):
    for j in range(i,n_task):    
        task_acc_vary[i][j]/=rp_times
    f.write("task %d: "%(i)+str(task_acc_vary[i][i:]))
    f.write("\n")
    plt.plot(list(range(i,n_task)), task_acc_vary[i][i:], c=seaborn.xkcd_rgb[c_list[i]], marker='X', label='task%d'%(i))

plt.legend()
plt.savefig(Img_path)
plt.show()

f.close()
