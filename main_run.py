import numpy as np
import os
import torch
from pairs_from_cps import LoadDataSet
from dataload import Pairwise_cp,Sequential_wind_loader,Triplet_pairwise_ldr
from torch.utils.data import   DataLoader
from models import  Linear_Sk_model
from distance import  SinkhornDistance
import  torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from  torch.optim.lr_scheduler import StepLR
from pathlib import Path
from change_det_methods import Change_detect
import  random
from os import listdir
from os.path import isfile, join
import glob
#data_series = Pairwise_cp(file_puath , no_of_pairs = 2)
#cp_pairs = data_series.get_segments_4m_cp(no_of_pairs = 2, buff = 0)


import torch
print(torch.__version__)
random.seed(2021)

#path of saved model
file_path_saved_nodel = './saved_model_paths/saved_SK_linear'


#set trainin ad testing paths
train_file_path = './data_files/simulated_data/GMM_switch/high_dim_changes_train_dim_100.mat'
test_file_list = './data_files/simulated_data/GMM_switch/high_dim_changes_test_dim_100.mat'


#the dimension of the projection (L in the paper)
dim_emb = 5

#regularization parameter (gamma in the paper)
eps = 0.1


#set window length
win_length = 10

#l_rate = 0.01
l_rate = 0.1
g_clip = 1

#reg_lambda (regularization for L1 regularization)
lamb  = 0


no_epochs = 2000

cp_paired_data_list = []
seq_windwd_data = np.asarray([])

#file path for saved model
file_path_saved_model = ''


train = 1 #set train or not train
load_model = 0
validate = 1

#get cp data, takes argument 2*window size to get sub-sequences
cp_data = LoadDataSet(train_file_path ,2*win_length,step_size = 25)

#number of triplet pairs pairs from each change point
n_pairs = 2

#get change points
cp_paired_data = cp_data.get_segments_4m_cp(no_of_pairs=n_pairs, buff=0)
cp_paired_data_list = cp_paired_data



train_ratio = 0.80
#get train ratio and validation splits
cp_paired_train = cp_paired_data_list[0: int(train_ratio * len(cp_paired_data_list))]
cp_paired_val = cp_paired_data_list[int(train_ratio * len(cp_paired_data_list)):]

#loader for triplet losses
cp_paired_train = Triplet_pairwise_ldr(cp_paired_train)
cp_paired_val = Triplet_pairwise_ldr(cp_paired_val)
#set test file path



cuda = 1

# doesn't really matter as training data is little. All the data can easily fit in a single batch.
batch_size = 512

#get number of f
input_size = cp_data.no_features


#Setting up the object for Sinkhorn distance
sk_dist = SinkhornDistance(eps= eps,max_iter = 200)

#defining the model for L
sk_lin_model = Linear_Sk_model(dim_emb=dim_emb,feat_size=input_size)


sk_lin_model.cuda()
input_size = cp_data.no_features

#setting the optimizer
optimizer_metric = optim.Adam(sk_lin_model.parameters(), lr= l_rate)

#Set triplet loss that takes in sinkhorn distance
trplet_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function= sk_dist , margin=1.0, swap=False, reduction='mean')

#set scheduler
scheduler = StepLR(optimizer_metric, step_size= 1000 , gamma=1)


def train_metric(pair_batch,l1_reg = 0):
    '''function trains L (projection matrix)...
    code takes in a batch of triplet pairs, along with L1_reg paramaeters'''
    loss_list =[]
    sk_lin_model.train()
    for param in sk_lin_model.parameters():
        param.requires_grad = True
    for i_batch, sample_batched in enumerate(pair_batch):
        optimizer_metric.zero_grad()

        #get triplets from batches
        X_anc = sample_batched['X_anc']
        X_dis = sample_batched['X_dissim']
        X_sim = sample_batched['X_sim']

        if cuda:
            X_anc, X_sim, X_dis = X_anc.cuda(0).float(), X_sim.cuda(0).float(), X_dis.cuda(0).float()
        X_anc_o = sk_lin_model(X_anc)
        X_sim_o = sk_lin_model(X_sim)
        X_dissim_o = sk_lin_model(X_dis)

        #compute triplet loss
        loss_trip = trplet_loss(X_anc_o, X_sim_o, X_dissim_o)

        #compute l1_reg cost
        loss_regu = torch.norm(torch.norm(sk_lin_model.linear_layer.weight, p=1, dim=1), p=1, dim=0)

        reg = l1_reg
        loss = loss_trip + reg * loss_regu
        loss.backward()
        optimizer_metric.step()
        loss_list.append(loss.item())

    return np.mean(np.asarray(loss_list))

def val_metric(pair_batch,l1_reg = 0):
    'comptutes validation loss'
    loss_list = []
    sk_lin_model.eval()


    'Batches for labeled and paired data'
    no_batches = len(pair_batch)

    for i_batch, sample_batched in enumerate(pair_batch):
        # print('Batch no {0}'.format(i_batch))

        optimizer_metric.zero_grad()

        #get triplet pairs
        X_anc = sample_batched['X_anc']
        X_dis = sample_batched['X_dissim']
        X_sim = sample_batched['X_sim']

        if cuda:
            X_anc, X_sim, X_dis = X_anc.cuda(0).float(), X_sim.cuda(0).float(), X_dis.cuda(0).float()
        X_anc_o = sk_lin_model(X_anc)
        X_sim_o = sk_lin_model(X_sim)
        X_dissim_o = sk_lin_model(X_dis)

       #computes loss

        loss = trplet_loss(X_anc_o, X_sim_o, X_dissim_o)

        loss_list.append(loss.item())

    return np.mean(np.asarray(loss_list))




train_btch = DataLoader(cp_paired_train, batch_size= batch_size, shuffle=False, drop_last=False)
val_btch = DataLoader(cp_paired_val, batch_size=batch_size, shuffle=False, drop_last=False)

if __name__ == "__main__":
        "Main loop"
        loss_val_best = 1e8

        'Load batches'
        train_btch = DataLoader(cp_paired_train, batch_size= 512, shuffle=False, drop_last=False)
        val_btch = DataLoader(cp_paired_val, batch_size=1000, shuffle=False, drop_last=False)


        if load_model == 1:
            sk_lin_model.load_state_dict(torch.load(file_path_saved_model))

        if train == 1:
                for ep in range(0,  no_epochs):

                        print('Labelled Epoch number: {0}'.format(ep))
                        loss_train =train_metric(pair_batch= train_btch)


                        loss_val = val_metric(pair_batch=train_btch)
                        print(loss_val)

                        scheduler.step()
                        if ep  > 10 and loss_train< loss_val_best:
                                torch.save(sk_lin_model.state_dict(), file_path_saved_model)
                                loss_val_best = loss_train
                                print(" : model saved at Epoch {0}".format(str(ep)))


        if validate == 1:
            #load test file
            cp_data_test = LoadDataSet(test_file_list, win_length)
            X = torch.from_numpy(cp_data_test.orig_signal)
            X_win_test = cp_data_test.X_windwd
            change_method = Change_detect('sink_horn_param', win_length=win_length, model=sk_lin_model)
            change_metric_enc = change_method.run_cp(X)
            change_metric_enc[0:win_length] = 0

            cp_labels = np.zeros((X.shape[0], 1))
            cp_labels[cp_data_test.sup_cp_indices] = 1
            y = cp_labels[0: - (len(cp_labels) - len(change_metric_enc))]
            fpr, tpr, thresholds = metrics.roc_curve(y, change_metric_enc)
            auc = metrics.auc(fpr, tpr)