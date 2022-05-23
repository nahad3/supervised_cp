'''This script experiments with neuro-rodent data'''

import numpy as np
from pairs_from_cps import LoadDataSet
from dataload import Pairwise_cp, Sequential_wind_loader, Triplet_pairwise_ldr
from torch.utils.data import random_split, DataLoader
from models import LSTM_model_VAE, LSTM_model, Linear_Sk_model
from distance import  SinkhornDistance, SinkhornDistance_Mahal, SinkhornDistance_Pytorch
import torch.optim as optim
import matplotlib.pyplot as plt
import mmd_util
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from change_det_methods import Change_detect
import random
from utils import ChangePointF1Score

import torch

print(torch.__version__)
random.seed(2022)

#set mode for training or load and running trained model
train = 0
load_and_test = 1


#select dataset (makes respective directories)
dataset = 'rodent_neural_slp_awake'


#path to save metric
file_path_save = './saved_model_paths/saved_SK_linear_neuro_rep' + dataset


#set learning parameters
train_ratio = 0.80
l_rate = 0.001
g_clip = 1
btch_size_trn = 128
#projection dimension of metric
dim_emb = 50
cp_paired_data_list = []
seq_windwd_data = np.asarray([])
no_epochs = 1000
reg = 0.5




#training file_set(contains change points that are used to learn a metric)
train_file_name = './../data/neural_rodent/sleep_awake_nerual_rec.mat'

#window length before and after a change that is split into smaller sub-sequences for similar-disimilar pairs for learning a metric
win_length = 200


'win_length is the segment length for downstream sinkhorn divergence'
win_length_cp = 100
#load trianing file. Input file along with change labels. File als
#step size determines the sliding window stride for getting sliding windows
cp_data = LoadDataSet(train_file_name, win_length, step_size=1)

#seq_windwd_data = np.concatenate((seq_windwd_data, cp_data.X_windwd), axis=0) if len(seq_windwd_data) \
#    else cp_data.X_windwd

seq_windwd_data = cp_data

#get number of pairs
n_pairs = 2
#buff: buffer aroud true change point
cp_paired_data = cp_data.get_segments_4m_cp(no_of_pairs=n_pairs,
                                            buff=50)  # returns a list of sim dissim pair

seq_windwd_data = cp_data.X_windwd
#incase multiple training files
seq_windwd_data = np.concatenate((seq_windwd_data, cp_data.X_windwd), axis=0) if len(seq_windwd_data) \
        else cp_data.X_windwd




#test file length
test_file_list = ['./../data/neural_rodent/slp_and_awake_nerual_rec.mat']


#if multiple datafiles
cp_paired_data_list = cp_paired_data
random.shuffle(cp_paired_data_list)
cp_paired_train = cp_paired_data_list[0: int(train_ratio * len(cp_paired_data_list))]
cp_paired_val = cp_paired_data_list[int(train_ratio * len(cp_paired_data_list)):]
torch.cuda.manual_seed(5)
np.random.seed(5)
torch.manual_seed(5)



seq_windwd_data_train = seq_windwd_data[0: int(train_ratio * len(seq_windwd_data)), :, :]
seq_windwd_data_val = seq_windwd_data[int(train_ratio * len(seq_windwd_data)):, :]
dloader_seq_win_train = Sequential_wind_loader(seq_windwd_data_train)

# get iterators
cp_paired_train = Triplet_pairwise_ldr(cp_paired_train)
cp_paired_val = Triplet_pairwise_ldr(cp_paired_val)

seg_length = cp_data.win_length / 2





cuda = 1
batch_size = 64

sk_dist = SinkhornDistance(eps=0.1, max_iter=200)
sk_dist_py = SinkhornDistance_Pytorch(eps=0.1,max_iter=200)
# defining the model


input_size = cp_data.no_features


sk_lin_model = Linear_Sk_model(dim_emb=dim_emb, feat_size=input_size)
sk_lin_model.cuda()


optimizer_metric = optim.Adam(sk_lin_model.parameters(), lr=l_rate)
trplet_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=sk_dist_py, margin= 0.1, swap=False,
                                                     reduction='mean')
scheduler = StepLR(optimizer_metric, step_size=1000, gamma=1)



def train_metric(pair_batch,reg=0.1):
    '''Function for training metric through CPs
     pair_batch: triplet batches, reg: regularization parametr for l1 norm'''

    loss_list = []
    sk_lin_model.train()
    for param in sk_lin_model.parameters():
        param.requires_grad = True

    'Batches for labeled and paired data'
    no_batches = len(pair_batch)
    #with torch.no_grad():
    #    #sk_lin_model.linear_layer.weight = nn.Parameter(torch.zeros_like(sk_lin_model.linear_layer.weight))
    #    #sk_lin_model.linear_layer.weight[0,5] = 1
    for i_batch, sample_batched in enumerate(pair_batch):
        # print('Batch no {0}'.format(i_batch))

        optimizer_metric.zero_grad()
        X_anc = sample_batched['X_anc']
        X_dis = sample_batched['X_dissim']
        X_sim = sample_batched['X_sim']

        if cuda:
            X_anc, X_sim, X_dis = X_anc.cuda(0).float(), X_sim.cuda(0).float(), X_dis.cuda(0).float()
        X_anc_o = sk_lin_model(X_anc)
        X_sim_o = sk_lin_model(X_sim)
        X_dissim_o = sk_lin_model(X_dis)


        loss_trip = trplet_loss(X_anc_o, X_sim_o, X_dissim_o)

        #norm for regularization (can be a mixed norm)
        loss_regu = torch.norm(torch.norm(sk_lin_model.linear_layer.weight, p=1, dim=1), p=2, dim=0)


        loss = loss_trip + reg * loss_regu
        loss.backward()

        optimizer_metric.step()
        loss_list.append(loss.item())


    return np.mean(np.asarray(loss_list))



def val_metric(pair_batch,reg=0.1):
    '''Function for validating learned metric through CPs
     pair_batch: triplet batches, reg: regularization parametr for l1 norm'''
    loss_list = []
    sk_lin_model.eval()


    'Batches for labeled and paired data'
    no_batches = len(pair_batch)

    for i_batch, sample_batched in enumerate(pair_batch):
        # print('Batch no {0}'.format(i_batch))

        optimizer_metric.zero_grad()
        X_anc = sample_batched['X_anc']
        X_dis = sample_batched['X_dissim']
        X_sim = sample_batched['X_sim']

        if cuda:
            X_anc, X_sim, X_dis = X_anc.cuda(0).float(), X_sim.cuda(0).float(), X_dis.cuda(0).float()
        X_anc_o = sk_lin_model(X_anc)
        X_sim_o = sk_lin_model(X_sim)
        X_dissim_o = sk_lin_model(X_dis)


        loss_trip = trplet_loss(X_anc_o, X_sim_o, X_dissim_o)
        loss_regu = torch.norm(torch.norm(sk_lin_model.linear_layer.weight, p=1, dim=1), p=2, dim=0)
        loss = loss_trip + reg * loss_regu
        loss_list.append(loss.item())

    return np.mean(np.asarray(loss_list))


if __name__ == "__main__":

    print("Training dataset: " + dataset)
    loss_val_best = 1e8
    train_btch = DataLoader(cp_paired_train, batch_size=btch_size_trn , shuffle=False, drop_last=False)
    val_btch = DataLoader(cp_paired_val, batch_size=btch_size_trn , shuffle=False, drop_last=False)



    if train == 1:
        for ep in range(0, no_epochs):

            print('Labelled Epoch number: {0}'.format(ep))
            loss_train = train_metric(pair_batch=train_btch)
            # loss_train = train_metric_no_cvx(pair_batch=train_btch)
            loss_val = val_metric(pair_batch=val_btch)
            print("Training loss:{0}".format(loss_train))
            print("Validation loss:{0}".format(loss_val))

            scheduler.step()
            if ep > 30 and loss_train < loss_val_best:
                torch.save(sk_lin_model.state_dict(), file_path_save)
                loss_val_best = loss_train
                print(dataset + " : model saved at Epoch {0}".format(str(ep)))

    elif load_and_test == 1:
        sk_lin_model.load_state_dict(torch.load(file_path_save))




    print("L matrix learned")
    L = sk_lin_model.linear_layer.weight
    # loss_train = train_metric(pair_batch=train_btch)
    L = L.detach().cpu().numpy()
    print(L)
    M = L.T @ L
    M[M<0]=0
    plt.matshow(M)
    plt.colorbar()
    plt.show()
    print("Mahb")
    label_plt = np.arange(1,43)
    fig, ax = plt.subplots()
    im = ax.imshow(abs(M), cmap='viridis')
    #plt.colorbar()
    ax.set_xticks(np.arange(0,42))
    ax.set_xticklabels(label_plt, size=5)
    ax.set_yticks(np.arange(0,42))
    ax.set_yticklabels(label_plt, size=5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="left",
             rotation_mode="anchor")



    plt.title("Learned Metric for Sleep stage", size=18)
    fig.colorbar(im,ax=ax)
    plt.show()
    #fig.savefig('./../figures/HASC/' + 'Rodent_metric.pdf')
    print(abs(M))
    # print(sk_lin_model.linear_layer.weight)
    if load_and_test == 1:
        "Run change points on learned metric"
        auc_list_enc = []
        auc_list_bline = []
        for n, test_path in enumerate(test_file_list):

            cp_data_test = LoadDataSet(test_path, int(30))

            X = torch.from_numpy(cp_data_test.signal)
            X_win_test = cp_data_test.X_windwd
            change_metric_enc = torch.zeros(X.shape[0])

            #change detection for SinkDiv
            change_method = Change_detect('sink_horn', win_length=win_length_cp ,eps = 1)
            change_metric = change_method.run_cp(X)
            # change detection for SinkDivLM (with learned metric)
            change_method = Change_detect('sink_horn_param', win_length=win_length_cp , model=sk_lin_model,eps = 1)
            change_metric_enc = change_method.run_cp(X)
            change_metric_enc[0:win_length_cp] = 0
            change_metric[0:win_length_cp] = 0
            cp_labels = np.zeros((X.shape[0], 1))
            cp_labels[cp_data_test.sup_cp_indices] = 1

            y = cp_labels[0: - (len(cp_labels) - len(change_metric))]

            fpr, tpr, thresholds = metrics.roc_curve(y, change_metric)
            auc_sinkdiv = metrics.auc(fpr, tpr)
            auc_list_bline.append(auc_sinkdiv)
            f1_sinnkdiv = ChangePointF1Score(y,15,fpr,tpr,thresholds,change_metric)

            y = cp_labels[0: - (len(cp_labels) - len(change_metric_enc))]
            y = cp_labels
            y = cp_labels[0: - (len(cp_labels) - len(change_metric_enc))]
            fpr, tpr, thresholds = metrics.roc_curve(y, change_metric_enc)
            # fpr, tpr, thresholds = metrics.roc_curve(y, d)
            auc_sinkdiv_lm = metrics.auc(fpr, tpr)
            auc_list_enc.append(auc_sinkdiv_lm)
            f1_enc = ChangePointF1Score(y,15,fpr,tpr,thresholds,change_metric_enc)
            visualize = 1
            if visualize == 1:
                f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
                # ax1.plot(X[:,:],label = 'Input sequence')
                ax1.plot(X[:, :])
                ax1.plot(cp_labels, label='Truth')
                ax1.set_title('{}-{} - Original signal'.format(dataset, str(n)))
                ax1.legend(loc=1, prop={'size': 6})

                ax2.plot(change_metric_enc, label='Change statistic')
                # ax2.plot(np.abs(cp_labels), label = 'Truth')
                ax2.vlines(cp_data_test.sup_cp_indices, ymin=0, ymax=np.max(change_metric_enc), color='red',
                           label='Truth')
                ax2.legend(loc=1, prop={'size': 6})

                ax2.set_title("Sinkhorn Divergence with learned ground metric")

                ax3.plot(change_metric, label="Change statistic")
                # ax3.plot(np.abs(cp_labels), label = 'Truth')
                ax3.vlines(cp_data_test.sup_cp_indices, ymin=0, ymax=np.max(change_metric), color='red', label='Truth')
                ax3.legend(loc=1, prop={'size': 6})
                ax3.set_title("Sinkhorn divergence (no learned ground metric)")
                plt.show()
        print("done")

np.round(np.mean(auc_list_enc), decimals=4)
np.round(np.mean(auc_list_bline), decimals=4)
print('here')