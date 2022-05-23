from torch import nn
import torch
from tcn import TemporalConvNet
import torch.nn.functional as F
import torch

class TCN(nn.Module):
    def __init__(self, input_size,  num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.sig = nn.Softmax(dim=2)
        self.nb = nn.BatchNorm1d(16)
    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        #output = self.nb(output)
        #output = F.normalize(output,dim = 2)
        #output = self.sig( output)
        return   output

class Feedforward(nn.Module):
    def __init__(self, input_size,  out_size,hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, hidden_size)
        self.fc6 = nn.Linear(self.hidden_size, out_size)

        self.sigmoid = nn.Sigmoid()
        self.sig = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden2 = self.fc2(relu)
        relu = self.relu(hidden2)
        hidden3 = self.fc3(relu)
        #relu = self.relu(hidden3)
       # hidden4 = self.fc4(relu)
       # relu = self.relu(hidden4)
        #hidden5 = self.fc5(relu)
       # relu = self.relu(hidden5)
       # hidden6 = self.fc5(relu)
       # relu = self.relu(hidden6)
        #output = self.fc6(relu)
        output = self.sig(hidden3)
        return output



class LSTM_model(nn.Module):
    'class for a K -1 lstm binary classifier (See diagram shared in email)'
    def __init__(self,  input_size, h_state_size,  lin_size, output_size):
        '''Structure of the network. LSTM followed by a linear layer, followed by relu. This s followed by anotherlinear layer
        that is the output layer. An empirical distribution is obtained over this output layer by applying hte softrmax operator'''

        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.h0 = 0
        self.lstm = nn.LSTM(input_size, h_state_size, batch_first=True)
        self.lstm2 = nn.LSTM(h_state_size,output_size, batch_first= True)
        self.relu1 = nn.ReLU()
        #self.batch_norm1 = nn.BatchNorm1d(lin_size)
        self.linear_layer = nn.Linear(h_state_size, lin_size)
        self.relu2 = nn.ReLU()
        #self.batch_norm2 = nn.BatchNorm1d(lin_size)
        self.final_linear   =  nn.Linear(lin_size, output_size)
        self.smax_layer = nn.Softmax(dim=0)


    def forward(self, x):
        batch_norm = 1
        self.h0 = torch.randn(1, 1, self.input_size)
        output, hidden = self.lstm(x)
        #output2, hidden2 = self.lstm2(output)

        lstm_f_state = output[:, - 1, :]






        lin_out = self.linear_layer(lstm_f_state)
        final_output =  self.final_linear(self.relu1(lin_out))
        final_output_norm  =  nn.functional.normalize(final_output , dim = 1)
        #if batch_norm:
        #    lin_out = self.batch_norm1(lin_out)
        #relu_lin_out = self.relu1(lin_out)
        #self.output_array = torch.empty(0,x.shape[0],2).cuda()
        #final_layer  = self.final_linear(relu_lin_out)
        #final_output = self.smax_layer(lin_out)





        return final_output_norm, output





class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size,  latent_length, dropout = 0, hidden_layer_depth = 1, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, batch_first=True)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        '''if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean, self.latent_logvar
        '''
        std = torch.exp(0.5 * self.latent_logvar)
        #std = torch.exp(0.5 *torch.tensor(0.00001))
        eps = torch.randn_like(std)
        #eps = torch.randn_like(0.0001)
        return eps.mul(std).add_(self.latent_mean), self.latent_mean,self.latent_logvar


class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self,  hidden_size,  latent_length, output_size, dtype, hidden_layer_depth = 1,block='LSTM'):
    #def __init__(self, sequence_length , batch_size,hidden_size, latent_length, output_size, dtype,
    #             hidden_layer_depth=1, block='LSTM'):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        #.batch_size = batch_size
        #self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth, batch_first=True)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_dept, batch_first=True)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        #self.decoder_inputs = torch.zeros( self.batch_size,self.sequence_length, 1, requires_grad=True).float().cuda()
        #self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).float().cuda()

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, x_shape ):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """

        self.decoder_inputs = torch.zeros( x_shape[0],x_shape[1], 1).float().cuda()
        self.c_0 = torch.zeros(self.hidden_layer_depth, x_shape[0], self.hidden_size).float().cuda()

        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            #h_0 = h_state
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class LSTM_model_VAE(nn.Module):
    def __init__(self, input_feat_size ,h_state_size, latent_length,output_size):
        super(LSTM_model_VAE, self).__init__()
        self.hidden_size = h_state_size
        self.latent_length = latent_length
        self.output_size = output_size
        self.input_feat_size = input_feat_size
        self.Encoder = Encoder(self.input_feat_size,self.hidden_size, self.latent_length)
        self.Lambda = Lambda(self.hidden_size, self.latent_length)
        #self.Decoder = Decoder(15,64,self.hidden_size, self.latent_length, self.output_size,float)
        self.Decoder = Decoder( self.hidden_size, self.latent_length, self.output_size, float)
    def forward(self,x):
        '''returns recon (reconstructed input) (z) hidden state. '''

        btch_size = x.shape[0]
        seq_length = x.shape[1]
        h_state = self.Encoder(x)
        z,mu,var = self.Lambda(h_state)
        recon = self.Decoder(z,(btch_size,seq_length))



        return recon, z,mu,var


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
