from torch import nn
import torch
import torch.nn.functional as F
import torch





class Linear_Sk_model(nn.Module):
    def __init__(self, dim_emb,feat_size):
        super(Linear_Sk_model, self).__init__()
        self.linear_layer = nn.Linear( feat_size,dim_emb,bias=False).float()
        self.relu1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(dim_emb,dim_emb).float()

    def forward(self,x):
        return self.linear_layer(x)
        #return  self.linear_layer2(self.relu1(self.linear_layer(x)))



class Non_Linear_Sk_model(nn.Module):
    def __init__(self, dim_emb,feat_size):
        super(Non_Linear_Sk_model, self).__init__()
        self.linear_layer = nn.Linear( feat_size,10,bias=False).float()
        self.relu1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(10,10).float()
        self.linear_layer3 = nn.Linear(10, dim_emb).float()
        self.relu2 = nn.ReLU()
    def forward(self,x):
        #return self.linear_layer(x)
        temp =  self.linear_layer2(self.relu1(self.linear_layer(x)))
        #return self.linear_layer3(self.relu2(temp))
        return temp
