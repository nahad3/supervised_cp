from torch.utils.data import Dataset
import numpy as np
import torch
import scipy.io as sio
from  itertools import  combinations as comb



class Sequential_lab_data(Dataset):
    def __init__(self,X,Y,win_length):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        X_windwd = X.unfold(0, win_length, 1)
        self.X_windwd = X_windwd.transpose(1, 2)
        Y_windwd = Y.unfold(0, win_length, 1)
        self.Y_windwd = Y_windwd.transpose(1, 2)
        self.Loc = np.zeros( (len(self.Y_windwd) ))
        for i in range(0,len(self.Y_windwd)):
            temp_arr = self.Y_windwd[i,:,:]
            if not len(np.where(temp_arr.reshape(-1) == 1)[0]):
                loc = -1
            else:
                loc = np.where(temp_arr.reshape(-1) == 1)[0][0]
            self.Loc[i] = loc

    def __len__(self):
        return self.X_windwd.shape[0]

    def __getitem__(self, item):

        X1 = self.X_windwd[item,:,:]
        Y1 = self.Y_windwd[item,:,:]
        L1 = self.Loc[item]
        sample = { 'X' :X1 , 'Y':Y1, 'L':L1 }

        return sample

class Pairwise_cp(Dataset):
    def __init__(self, cp_pair_list):
        '''cp_pair_list: list containting data for cp_pairs. Each item in this list contains sim dissim pairs generated \
        from a change point. Thus number of items in this list = no of change points.
        Output of get_segments_4m_cp function in LoadData class'''
        self.X1 = np.asarray([])
        self.X2 = np.asarray([])
        self.Y = np.asarray([])

        self.X1_s = np.asarray([])
        self.X2_s = np.asarray([])
        self.X1_d = np.asarray([])
        self.X2_d = np.asarray([])

        self.X_anc = np.asarray([])
        self.X_sim = np.asarray([])
        self.X_dissim = np.asarray([])

        self.cp_pair_list = cp_pair_list
        self.unroll_paired_data()
        print("here")

    def unroll_paired_data(self):
        '''This function takes in the cp_pair list provided to this function and unrolls it to get a large number of sim \
        dissim pairs. This would help with getting batches '''
        one_np = np.asarray(1).reshape(-1,1)
        for cp_data in self.cp_pair_list:
            X1_d_temp = cp_data['X1_d']
            X2_d_temp = cp_data['X2_d']
            X1_s_temp = cp_data['X1_s']
            X2_s_temp = cp_data['X2_s']

            diff_no = X1_d_temp.shape[0]
            same_no = X1_s_temp.shape[0]
            same_lab = np.ones(same_no).reshape(-1, 1)
            diff_lab = -np.ones(diff_no).reshape(-1, 1)


            self.X1 = np.concatenate((self.X1, X1_s_temp), axis=0) if self.X1.size \
                else X1_s_temp
            self.X2 = np.concatenate((self.X2, X2_s_temp), axis=0) if self.X2.size \
                else X2_s_temp
            self.Y = np.concatenate((self.Y, same_lab ), axis = 0 ) if self.Y.size \
                else same_lab
            self.X1 = np.concatenate((self.X1, X1_d_temp), axis=0)
            self.X2 = np.concatenate((self.X2, X2_d_temp), axis=0)

            self.Y = np.concatenate((self.Y, diff_lab) ,axis = 0)

            self.X_anc = np.concatenate((self.X_anc, X1_s_temp), axis=0) if self.X_anc.size \
                else X1_s_temp
            self.X_sim = np.concatenate((self.X_sim, X2_s_temp), axis=0) if self.X_sim \
                else X2_s_temp
            self.X_dissim = np.concatenate((self.X_dissim, X2_d_temp), axis=0) if self.X_dissim \
                else X2_d_temp


            self.X1_s = np.concatenate((self.X1_s))
    def __getitem__(self, idx):
        X1= self.X1[ idx ]
        X2= self.X2[idx]
        Y = self.Y[idx]




        sample = {'X1': X1, 'X2': X2, 'Y': Y }
        return sample

    def __len__(self):
        return len(self.X1)


class Triplet_pairwise_ldr(Dataset):
    def __init__(self, cp_pair_list):
        '''cp_pair_list: list containting data for cp_pairs. Each item in this list contains sim dissim pairs generated \
        from a change point. Thus number of items in this list = no of change points.
        Output of get_segments_4m_cp function in LoadData class'''
        self.X1 = np.asarray([])
        self.X2 = np.asarray([])
        self.Y = np.asarray([])

        self.X1_s = np.asarray([])
        self.X2_s = np.asarray([])
        self.X1_d = np.asarray([])
        self.X2_d = np.asarray([])

        self.X_anc = np.asarray([])
        self.X_sim = np.asarray([])
        self.X_dissim = np.asarray([])

        self.cp_pair_list = cp_pair_list
        self.unroll_paired_data()
        print("here")

    def unroll_paired_data(self):
        '''This function takes in the cp_pair list provided to this function and unrolls it to get a large number of sim \
        dissim pairs. This would help with getting batches '''
        one_np = np.asarray(1).reshape(-1, 1)
        for cp_data in self.cp_pair_list:
            X1_d_temp = cp_data['X1_d']
            X2_d_temp = cp_data['X2_d']
            X1_s_temp = cp_data['X1_s']
            X2_s_temp = cp_data['X2_s']

            diff_no = X1_d_temp.shape[0]
            same_no = X1_s_temp.shape[0]
            same_lab = np.ones(same_no).reshape(-1, 1)
            diff_lab = -np.ones(diff_no).reshape(-1, 1)

            self.X1 = np.concatenate((self.X1, X1_s_temp), axis=0) if self.X1.size \
                else X1_s_temp
            self.X2 = np.concatenate((self.X2, X2_s_temp), axis=0) if self.X2.size \
                else X2_s_temp
            self.Y = np.concatenate((self.Y, same_lab), axis=0) if self.Y.size \
                else same_lab
            self.X1 = np.concatenate((self.X1, X1_d_temp), axis=0)
            self.X2 = np.concatenate((self.X2, X2_d_temp), axis=0)

            self.Y = np.concatenate((self.Y, diff_lab), axis=0)

            self.X_anc = np.concatenate((self.X_anc, X1_s_temp), axis=0) if self.X_anc.size \
                else X1_s_temp
            self.X_sim = np.concatenate((self.X_sim, X2_s_temp), axis=0) if self.X_sim.size \
                else X2_s_temp
            self.X_dissim = np.concatenate((self.X_dissim, X2_d_temp), axis=0) if self.X_dissim.size \
                else X2_d_temp

            #self.X1_s = np.concatenate((self.X1_s))

    def __getitem__(self, idx):
        X_anc = self.X_anc[idx]
        X_sim = self.X_sim[idx]
        X_dissim = self.X_dissim[idx]

        sample = {'X_anc': X_anc, 'X_sim': X_sim, 'X_dissim': X_dissim}
        return sample

    def __len__(self):
        return len(self.X_anc)

class Pairwise_cp_add_feats(Dataset):
    def __init__(self, cp_pair_list):
        '''cp_pair_list: list containting data for cp_pairs. Each item in this list contains sim dissim pairs generated \
        from a change point. Thus number of items in this list = no of change points.
        Output of get_segments_4m_cp function in LoadData class'''
        self.X1 = np.asarray([])
        self.X2 = np.asarray([])
        self.Y = np.asarray([])
        self.cp_pair_list = cp_pair_list
        self.unroll_paired_data()
        print("here")

    def unroll_paired_data(self):
        '''This function takes in the cp_pair list provided to this function and unrolls it to get a large number of sim \
        dissim pairs. This would help with getting batches '''
        one_np = np.asarray(1).reshape(-1,1)
        for cp_data in self.cp_pair_list:
            X1_d_temp = cp_data['X1_d']
            X2_d_temp = cp_data['X2_d']
            X1_s_temp = cp_data['X1_s']
            X2_s_temp = cp_data['X2_s']

            diff_no = X1_d_temp.shape[0]
            same_no = X1_s_temp.shape[0]
            same_lab = np.ones(same_no).reshape(-1, 1)
            diff_lab = -np.ones(diff_no).reshape(-1, 1)


            self.X1 = np.concatenate((self.X1, X1_s_temp), axis=0) if self.X1.size \
                else X1_s_temp
            self.X2 = np.concatenate((self.X2, X2_s_temp), axis=0) if self.X2.size \
                else X2_s_temp
            self.Y = np.concatenate((self.Y, same_lab ), axis = 0 ) if self.Y.size \
                else same_lab
            self.X1 = np.concatenate((self.X1, X1_d_temp), axis=0)
            self.X2 = np.concatenate((self.X2, X2_d_temp), axis=0)

            self.Y = np.concatenate((self.Y, diff_lab) ,axis = 0)


    def __getitem__(self, idx):
        X1= self.X1[ idx ]
        X2= self.X2[idx]
        Y = self.Y[idx]




        sample = {'X1': X1, 'X2': X2, 'Y': Y }
        return sample

    def __len__(self):
        return len(self.X1)

class Sequential_wind_loader(Dataset):
    def __init__(self, windowd_data):
        self.windwd_data = windowd_data

    def __len__(self):
        return self.windwd_data.shape[0]

    def __getitem__(self, item):

        X = self.windwd_data[item,:,:]

        return self.windwd_data[item,:,:]