import numpy as np
import torch
from distance import Energy_Distance, SinkhornDistance,SinkhornDistance_Mahal, Window_gauss_KL as KL
import time

class Change_detect():
    def __init__(self,change_stat,win_length,model= 0 ,M=0,eps=1):
        self.win_length = win_length
        self.change_methd_name = change_stat
        self.model = model
        self.M = M
        self.eps = eps
        if change_stat == 'sink_horn_param' :
            assert self.model != 0

        #change_methods = {
        #    'kl_div' : KL(),
        #    'sink_horn' : SinkhornDistance(eps= 1, max_iter=200),
        #    'sink_horn_param': SinkhornDistance(eps= 1, max_iter=200),
        #    'sink_horn_Mahal' : SinkhornDistance_Mahal(eps= 1,max_iter = 200)
        #}

        change_methods = {
            'kl_div': KL(),
            'sink_horn': SinkhornDistance(eps= self.eps, max_iter=200),
            'sink_horn_param': SinkhornDistance(eps= self.eps, max_iter=200),
            'sink_horn_Mahal': SinkhornDistance_Mahal(eps=self.eps, max_iter=200)
        }
        self.change_stat = change_methods[change_stat]
    def run_cp(self,X):
        X = X.unfold(0, self.win_length, 1).transpose(1, 2)
        X1 = X[0:-self.win_length, :, :]
        X2 = X[self.win_length:, :, :]
        if self.change_methd_name == 'sink_horn_param':


            self.model =self.model.cuda().float()

            self.model.eval()
            change_stat = np.zeros(X1.shape[0]-1)
            btch_size = 90
            for i in range(0,len(X1) -btch_size,btch_size):
                torch.cuda.empty_cache()
                X1_s = X1[i:i + btch_size,:,:]
                X2_s = X2[i: i +btch_size,:,:]
                X1_enc = self.model(X1_s.cuda().detach().float())
                X2_enc = self.model(X2_s.cuda().detach().float())
                change_stat[i:i+btch_size] = self.change_stat(X1_enc,X2_enc).detach().cpu().numpy() -0.5*(self.change_stat(X2_enc,X2_enc).detach().cpu().numpy() + self.change_stat(X1_enc,X1_enc).detach().cpu().numpy())
                #change_stat[i:i + btch_size] = self.change_stat(torch.from_numpy(X1_s).cuda().float(), torch.from_numpy(X2_s).cuda().float()).detach().cpu().numpy()
                X1_enc = 0
                X2_enc =0
                torch.cuda.empty_cache()
                print(i)

            change_stat = np.roll(change_stat, self.win_length)
            return change_stat

        if self.change_methd_name == 'sink_horn_Mahal':
            assert (self.M !=0,'Please provide M')
            M = torch.from_numpy(self.M).cuda().float()
            change_stat = np.zeros(X1.shape[0]-1)
            btch_size = 90
            for i in range(0,len(X1) -btch_size,btch_size):
                torch.cuda.empty_cache()
                X1_s = X1[i:i + btch_size,:,:].cuda().float()
                X2_s = X2[i: i +btch_size,:,:].cuda().float()

                change_stat[i:i+btch_size] = self.change_stat(X1_s,X2_s,M)[0].detach().cpu().numpy() -0.5*(self.change_stat(X1_s,X1_s,M)[0].detach().cpu().numpy() + self.change_stat(X2_s,X2_s,M)[0].detach().cpu().numpy())
                #change_stat[i:i + btch_size] = self.change_stat(torch.from_numpy(X1_s).cuda().float(), torch.from_numpy(X2_s).cuda().float()).detach().cpu().numpy()
                X1_enc = 0
                X2_enc =0
                torch.cuda.empty_cache()
                print(i)

            change_stat = np.roll(change_stat, self.win_length)
            return change_stat

        if self.change_methd_name == 'sink_horn':



            change_stat = np.zeros(X1.shape[0] - 1)
            btch_size = 90
            for i in range(0, len(X1) - btch_size, btch_size):
                print (i)
                torch.cuda.empty_cache()
                X1_s = X1[i: i +btch_size, :, :].unsqueeze(-1).cuda().double().squeeze(-1)
                X2_s = X2[i: i + btch_size, :, :].unsqueeze(-1).cuda().double().squeeze(-1)
                change_stat[i:i + btch_size] = self.change_stat(X1_s, X2_s).detach().cpu().numpy() - 0.5*(self.change_stat(X1_s, X1_s).detach().cpu().numpy() + self.change_stat(X2_s, X2_s).detach().cpu().numpy())
                torch.cuda.empty_cache()
            change_stat = np.roll(change_stat,self.win_length)
            return change_stat

        else:
            change_stat = np.zeros(X.shape[0]-1)
            for i in range(self.win_length,len(X) - self.win_length):
                x_1 = X[i- self.win_length: i ,:]
                x_2 = X[i : i + self.win_length,:]

                change_stat[i] = self.change_stat(x_1.unsqueeze(0),x_2.unsqueeze(0)) + self.change_stat(x_2.unsqueeze(0),x_1.unsqueeze(0))
        return change_stat