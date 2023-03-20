import os
import numpy as np
import scipy.io as sio
from  itertools import  combinations as comb
from sklearn import preprocessing
import torch
import random
from data_aug.data_aug_lib import DA_FlipLR
import matplotlib.pyplot as plt

class LoadDataSet():
    def __init__(self, file_path, seg_length = None, step_size=1):
        '''signal: Signal on which CP to be detected
        sup_cp_i: indicies for change points (provided in a supervised manner)
        pair_length: length of each segment in similar/dissimilar pairs returned
        buffer: buffer before and after c efore obtaining pair'''
        self.mat_file = sio.loadmat(file_path)
        self.signal = np.array(self.mat_file['Y'])
        self.orig_signal = self.signal
        self.no_features = self.signal.shape[1]
        'scaling each axis between 0 and 1'
        min_max_scaler = preprocessing.MinMaxScaler()
        self.signal = min_max_scaler.fit_transform(self.signal)
        self.sup_cp_indices = np.array(self.mat_file['L']).reshape(-1,)



        self.step_size = step_size
        if np.shape(self.sup_cp_indices)[0] == self.signal.shape[0]:
            self.sup_cp_indices = np.where(self.sup_cp_indices == 1)[0]


        window_block = np.diff(self.sup_cp_indices)
        window_block = np.min(window_block)
        if window_block % 2 != 0:
            window_block = window_block - 1
        self.win_length = window_block

        if seg_length == None:
            self.seg_length = window_block
        else:
            self.seg_length = seg_length

        self.cp_labels = np.zeros((self.signal.shape[0], 1))
        self.cp_labels[self.sup_cp_indices] = 1
        X = torch.from_numpy(self.signal)
        X_windwd = X.unfold(0,int(self.seg_length),1)
        self.X_windwd = np.zeros( (self.signal.shape[0]-1,self.seg_length,self.signal.shape[-1]))
        temp = X_windwd.transpose(1, 2)
        self.X_windwd[len(self.X_windwd) - len(temp):, :, :] = temp


    def get_segments_4m_cp(self,   no_of_pairs, buff =0,aug = 0):
        '''window block: A big window size before and after the cp from where smaller segments of sim
        dissim pairs are extracted. (should be large enough to get 2 pairs)
        pair_length: length of smaller segments extracted from larger window black

        '''

        signal = self.signal
        cp_indices = self.sup_cp_indices.tolist()

        # seg_size has to be even

        'window block determined from consecutive cps but could be changed later'

        sig_dim = signal.shape[-1]
        X1 = np.array([])
        X2 = np.array([])

        #self.win_length = 60
        if self.win_length == None:
            window_block = np.diff(self.sup_cp_indices)
            window_block = np.min(window_block)
            if window_block % 2 != 0:
                window_block = window_block - 1
            self.win_length = window_block

        else:
            window_block = self.win_length



        if window_block % 2 != 0:
            window_block =  window_block -1
        #seg_size = seg_size if seg_size % 2 == 0 else seg_size - 1




        #if signal.shape[0] - cp_indices[-1] - buff < 2 * seg_size: cp_indices = cp_indices[:-1]



        cp_pairs_list = []

        for cp in cp_indices:  ## For non yahoos cp_indices[:-2]:

            'Checking if sufficient length available for last cp to get segment after cp'
            X_p = signal[cp  - self.seg_length - buff : cp - buff , :]
            X_f = signal[ cp + buff : cp  +    self.seg_length + buff , :]
            if len(X_p) == len(X_f):
                cp_pairs = self.get_pairs(X_p, X_f)
                if cp_pairs == -1:
                    continue

                else:
                    cp_pairs_list.append(cp_pairs)

                aug = 1
                if aug == 1:
                    X_p_LR = DA_FlipLR(X_p)
                    X_f_LR = DA_FlipLR(X_f)
                    X_p_UD = DA_FlipLR(X_p)
                    X_f_UD = DA_FlipLR(X_f)

                    cp_pairs_UD = self.get_pairs(np.ascontiguousarray(X_p_UD),np.ascontiguousarray(X_f_UD))
                    'can add more here'
                    print("here")

                    '''
                    plt.plot(X_p[:, 5])
                    plt.plot(X_f[:, 5])
                    plt.show()
                    
                    '''
                    cp_pairs_list.append(cp_pairs_UD)

                    cp_pairs_temp = {}
                    cp_pairs_temp['X1_s'] = cp_pairs['X1_s']
                    cp_pairs_temp['X2_s'] = cp_pairs_UD['X1_s']
                    cp_pairs_temp['X1_d'] = cp_pairs['X1_s']
                    cp_pairs_temp['X2_d'] = cp_pairs_UD['X2_d']
                    cp_pairs_list.append(cp_pairs_temp)

        return cp_pairs_list

    def get_pairs(self,X_p,X_f, no_of_pairs =2):

        X1_s = np.array([])
        X2_s = np.array([])

        X1_d = np.array([])
        X2_d = np.array([])



        if self.win_length > self.seg_length:
            '''if size of win length  is greater than self.get length (consecutive segments) then divide X_p into smaller 
            segments of size seg_length . Do what was done below'''
            no_of_wins = int(self.win_length / self.seg_length)
            if no_of_wins > 1:
                print("here")
            else:
                X_p = X_p[ - self.seg_length  : , :]
                X_f = X_f[ : self.seg_length , :]

        else:
            no_of_augs = int(self.seg_length / self.win_length)

            while len(X_p) < self.seg_length:
                X_p_o = X_p
                X_f_o = X_f
                X_p_flp = DA_FlipLR(X_p_o)
                X_f_flp = DA_FlipLR(X_f_o)
                print("Here")
                X_p = np.concatenate((X_p_flp, X_p), axis=0)
                X_f = np.concatenate((X_f, X_f_flp), axis=0)

            X_p = X_p[-self.seg_length:, :]
            X_f = X_f[:self.seg_length, :]
            print('here')

        # seg_size,_  = np.shape(X_p_s[0])

        X_p_s = np.split(X_p, no_of_pairs)
        X_f_s = np.split(X_f, no_of_pairs)

        '''
        Xp_windwd = torch.from_numpy(X_p).unfold(0, int(self.seg_length / 2), self.step_size)
        Xp_windwd = Xp_windwd.transpose(1, 2)

        Xf_windwd = torch.from_numpy(X_f).unfold(0, int(self.seg_length / 2), self.step_size)
        Xf_windwd = Xf_windwd.transpose(1, 2)

        Xf1 = Xf_windwd.repeat_interleave(Xf_windwd.shape[0], 0)
        Xf2 = Xf_windwd.repeat(Xf_windwd.shape[0], 1, 1)

        Xp1 = Xp_windwd.repeat_interleave(Xp_windwd.shape[0], 0)
        Xp2 = Xp_windwd.repeat(Xp_windwd.shape[0], 1, 1)

        X1_s = np.concatenate((X1_s, Xf1), axis=0) if X1_s.size \
            else Xf1
        X2_s = np.concatenate((X2_s, Xf2), axis=0) if X2_s.size \
            else Xf2

        X1_s = np.concatenate((X1_s, Xp1), axis=0) if X1_s.size \
            else Xp1
        X2_s = np.concatenate((X2_s, Xp2), axis=0) if X2_s.size \
            else Xp2

        X1_d = np.concatenate((X1_d, Xp1), axis=0) if X1_d.size \
            else Xp1
        X2_d = np.concatenate((X2_d, Xf2), axis=0) if X2_d.size \
            else Xf2

        X1_d = np.concatenate((X1_d, Xp2), axis=0) if X1_d.size \
            else Xp2
        X2_d = np.concatenate((X2_d, Xf1), axis=0) if X2_d.size \
            else Xf2

        number_samples = min(X1_s.shape[0], X1_s.shape[0])
        rand_integers = random.sample(range(0, X1_s.shape[0]), number_samples)
        # X1_s =  X1_s[rand_integers,:,:]
        # X2_s = X2_s[rand_integers,: ,:]

        # X1_d = X1_d[rand_integers,:,:]
        # X2_d = X2_d[rand_integers,:,:]

        cp_pairs = {}
        #cp_pairs['index'] = cp
        cp_pairs['X1_d'] = X1_d
        cp_pairs['X2_d'] = X2_d
        cp_pairs['X1_s'] = X1_s
        cp_pairs['X2_s'] = X2_s

        if len(X1_d) != len(X2_d):
            return -2

        if len(X1_s) != len(X2_s):
            print('here')
            return  -2

        return cp_pairs
        '''
        #seg_size = int(self.seg_length/2)

        'seg size after splitting'
        seg_size = X_p_s[0].shape[0]

        for i in range(0,len(X_p_s)):
            X1_d = np.concatenate((X1_d, X_p_s[i].reshape(1, seg_size, -1)), axis=0) if X1_d.size \
                else X_p_s[i].reshape(1, seg_size, -1)
            try:
                X2_d = np.concatenate((X2_d, X_f_s[i].reshape(1, seg_size, -1 )), axis=0) if X2_d.size \
                else X_f_s[i].reshape(1, seg_size, -1)
            except ValueError:
                print('here')
      

        comb_list = list(comb(np.arange(0, no_of_pairs), 2))

        for pair in comb_list:
            'get sim pairs from past window'
            X1p_temp = X_p_s[pair[0]].reshape(1, seg_size, -1)
            X2p_temp = X_p_s[pair[1]].reshape(1, seg_size, -1)
            X1_s = np.concatenate((X1_s, X1p_temp), axis=0) if X1_s.size \
                else X1p_temp
            X2_s = np.concatenate((X2_s, X2p_temp), axis=0) if X2_s.size \
                else X2p_temp



            'get sim pairs from fut window'
            X1f_temp = X_f_s[pair[0]].reshape(1, seg_size, -1)
            X2f_temp = X_f_s[pair[1]].reshape(1, seg_size, -1)
            X1_s = np.concatenate((X1_s, X1f_temp), axis=0) if X1_s.size \
                else X1f_temp
            X2_s = np.concatenate((X2_s, X2f_temp), axis=0) if X2_s.size \
                else X2f_temp

            #X1_d = np.concatenate((X1_d, X1p_temp), axis=0) if X1_d.size \
            #   else X1p_temp
            #X2_d = np.concatenate((X2_d, X2f_temp), axis=0) if X2_d.size \
            #    else X2f_temp

            X1_d = np.concatenate((X1_d, X2p_temp), axis=0) if X1_d.size \
                else X2p_temp
            X2_d = np.concatenate((X2_d, X1f_temp), axis=0) if X2_d.size \
                else X1f_temp

            X1_d = np.concatenate((X1_d, X1p_temp), axis=0) if X1_d.size \
                else X1p_temp
            X2_d = np.concatenate((X2_d, X1f_temp), axis=0) if X2_d.size \
                else X1f_temp

            #X1_d = np.concatenate((X1_d, X2p_temp), axis=0) if X1_d.size \
            #    else X2p_temp
            #X2_d = np.concatenate((X2_d, X2f_temp), axis=0) if X2_d.size \
            #    else X1f_temp

        cp_pairs = {}
        #cp_pairs['index'] = cp
        cp_pairs['X1_d']  = X1_d
        cp_pairs['X2_d']  = X2_d
        cp_pairs['X1_s'] = X1_s
        cp_pairs['X2_s'] = X2_s

        return cp_pairs




