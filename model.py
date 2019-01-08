import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNNmodel(nn.Module):
    def __init__(self,input_size, batch_size,hidden_size, n_layers=1, dropout=0.0):
        super(EncoderRNNmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,dropout = dropout,bidirectional=False)
       
    def forward(self,input_tensor):
        c0,h0 = self.init_hidden()
        packed_outputs,states = self.lstm(input_tensor,(c0,h0))
        return states
    def init_hidden(self):
        c0 = torch.randn(self.n_layers,self.batch_size,self.hidden_size,requires_grad = False).to(device)
        h0 = torch.randn(self.n_layers,self.batch_size,self.hidden_size,requires_grad = False).to(device)
        return c0,h0
    
class DecoderRNNmodel(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1,dropout=0.0):
        super(DecoderRNNmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,dropout=dropout,bidirectional=False)
        self.linear = nn.Linear(hidden_size,4,bias=True)
    def forward(self,input_tensor,hidden_states,mode):
        if mode=="train":
            outputs,_ =  self.lstm(input_tensor,hidden_states)
            outputs = outputs.view(-1,self.hidden_size)
            outputs = self.linear(outputs)
            outputs[:,:-2] = F.tanh(outputs[:,:-2].clone())
            outputs[:,2:] = F.sigmoid(outputs[:,2:].clone())
            return outputs
        else:
            outputs = []
            eop = []
            input_tensor = input_tensor[0]
            input_tensor = input_tensor.view(1,1,4)
            i=0
            while(input_tensor[0,0,3]<=0.5):
                input_tensor[0,0,3]=0.0
                input_tensor[0,0,2]=0.0
                output,hidden_states = self.lstm(input_tensor,hidden_states)
                output = output.view(-1,self.hidden_size)
                output = self.linear(output)
                output[:,:-2] = F.tanh(output[:,:-2].clone())
                output[:,2:] = F.sigmoid(output[:,2:].clone())
                outputs.append(output)
                input_tensor = output.view(1,1,self.input_size)
                del output
                i+=1
                if i>200:
                    break
            return torch.cat(outputs)
class Model(nn.Module):
    def __init__(self,en_input_size,batch_size,en_hidden_size,de_hidden_size,en_layers,de_layers,en_dropout,de_dropout):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.de_hidden_size = de_hidden_size
        self.en_input_size = en_input_size
        self.enc = EncoderRNNmodel(en_input_size,batch_size,en_hidden_size,en_layers,en_dropout)
        self.dec = DecoderRNNmodel(en_input_size,de_hidden_size,de_layers,de_dropout)
    def forward(self,input_tensor,input_tensor2,mode):
        en_states = self.enc(input_tensor)
        #here we can change the en_outputs and en_states in desired shape
        c,h = en_states
        c = c.view(1,self.batch_size,self.de_hidden_size)
        h = h.view(1,self.batch_size,self.de_hidden_size)
        input_tensor2 = input_tensor2.view(-1,self.batch_size,self.en_input_size)
        en_states = (c,h)
        output = self.dec(input_tensor2,en_states,mode)
        return output

