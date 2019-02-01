import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import Image_classifier

class SkipFeature(nn.Module):
    def __init__(self,concat_channels=8):
        super(SkipFeature, self).__init__()

        # Default transform for all torchvision models
    
        self.resnet = Image_classifier.Image_classifier(64,832,1)
        #input for concat1 has dimension = 60*60*13
        
        concat1 = nn.Conv2d(13, concat_channels, kernel_size=6,stride=2, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        #outputs 28*28*concat_channels
        
        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)
        #input for concat2 has dimension = 28*28*5
        concat2 = nn.Conv2d(13, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        #outputs 28*28*concat_channels
        
        self.res1_concat = nn.Sequential(concat2, bn2, relu2)
        #input for concat3 has dimension = 12*12*5
        concat3 = nn.Conv2d(5, concat_channels, kernel_size=3, padding=2, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        #outputs 28*28*concat_channels
        
        self.res2_concat = nn.Sequential(concat3, bn3, relu3, up3)
        #input for concat4 has dimension = 4*4*5
        concat4 = nn.Conv2d(5, concat_channels, kernel_size=2, padding=2, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        #outputs 28*28*concat_channels
        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)

        # Different from original, original used maxpool
        # Original used no activation here
        #all inputs are of dimension 28*28, we need to convert them to 8*8
        conv_final_1 = nn.Conv2d(4*concat_channels, 16, kernel_size=6,stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(16)
        conv_final_2 = nn.Conv2d(16, 1, kernel_size=5, bias=False)
        bn_final_2 = nn.BatchNorm2d(1)

        self.conv_final = nn.Sequential(conv_final_1, bn_final_1,conv_final_2, bn_final_2)
        #self.linear = nn.Linear(20,64,bias=True)
    def reload(self, path):
        print("Reloading resnet from: ", path)
        self.resnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['state_dict'])

    def forward(self, x):
        x = x.view(-1,1,64,832)
        conv1_f, layer1_f, layer2_f,layer4_f,fc = self.resnet(x)
        conv1_f = self.conv1_concat(conv1_f)
        layer1_f = self.res1_concat(layer1_f)
        layer2_f = self.res2_concat(layer2_f)
        layer4_f = self.res4_concat(layer4_f)

        concat_features = torch.cat((conv1_f, layer1_f, layer2_f, layer4_f), dim=1)
        #final_features = self.linear(fc)
        final_features = self.conv_final(concat_features)
        final_features = final_features.view(-1,64)
        return final_features


class EncoderRNNmodel(nn.Module):
    def __init__(self,input_size, batch_size,hidden_size, n_layers=1, dropout=0.0):
        super(EncoderRNNmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,dropout = dropout,bidirectional=False)
        self.linear = nn.Linear(hidden_size,input_size+1,bias=True)
    def forward(self,input_tensor,hidden_tensor):
        c0,h0 = hidden_tensor.view(self.n_layers,-1,self.hidden_size), hidden_tensor.view(self.n_layers,-1,self.hidden_size)
        outputs,(cn,hn) = self.lstm(input_tensor,(c0,h0))
        output = outputs[-1]
        output = self.linear(output)
        output = F.sigmoid(output.clone())
        cn = F.tanh(cn)
        hn = F.tanh(hn)
        return output,(cn,hn)
    def init_hidden(self):
        c0 = torch.zeros(self.n_layers,self.batch_size,self.hidden_size).to(device)
        h0 = torch.zeros(self.n_layers,self.batch_size,self.hidden_size).to(device)
        return c0,h0
    
class DecoderRNNmodel(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1,dropout=0.0):
        super(DecoderRNNmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,dropout=dropout,bidirectional=False)
        self.linear = nn.Linear(hidden_size,3,bias=True)
    def forward(self,input_tensor,hidden_states,mode):
        if mode=="train":
            outputs,_ =  self.lstm(input_tensor,hidden_states)
            outputs = outputs.view(-1,self.hidden_size)
            outputs = self.linear(outputs)
            outputs = F.sigmoid(outputs.clone())
            return outputs
        else:
            outputs = []
            eop = []
            input_tensor = input_tensor.view(1,1,3)
            i=0
            while(input_tensor[0,0,2]<=0.95):
                output,hidden_states = self.lstm(input_tensor,hidden_states)
                output = output.view(-1,self.hidden_size)
                output = self.linear(output)
                output = F.sigmoid(output.clone())
                outputs.append(output)
                input_tensor = output.view(1,1,self.input_size)
                del output
                i+=1
                if i>100:
                    break
            return torch.cat(outputs)
class Model(nn.Module):
    def __init__(self,en_input_size,dec_input_size,batch_size,en_hidden_size,de_hidden_size,en_layers,de_layers,en_dropout,de_dropout):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.de_hidden_size = de_hidden_size
        self.en_input_size = en_input_size
        self.dec_input_size = dec_input_size
        self.enc = EncoderRNNmodel(en_input_size,batch_size,en_hidden_size,en_layers,en_dropout)
        self.dec = DecoderRNNmodel(dec_input_size,de_hidden_size,de_layers,de_dropout)
        self.image_encoder = SkipFeature()
        
    def forward(self,img,input_tensor,input_tensor2,mode):
        hidden_tensor = self.image_encoder(img)
        input_tensor = input_tensor.view(-1,self.batch_size,self.en_input_size)
        en_output,en_states = self.enc(input_tensor,hidden_tensor)
        #here we can change the en_outputs and en_states in desired shape
        #hidden_tensor = hidden_tensor.view(1,-1,64)
        c,h = en_states
        c = c.view(1,self.batch_size,self.de_hidden_size)
        h = h.view(1,self.batch_size,self.de_hidden_size)
        #c = torch.cat((c,hidden_tensor),dim=2)
        #h = torch.cat((h,hidden_tensor),dim=2)
        
        input_tensor2 = input_tensor2.view(-1,self.batch_size,self.dec_input_size)
        en_states = (c,h)
        if mode == 'train':
            output = self.dec(input_tensor2,en_states,mode)
        else:
            output = self.dec(en_output,en_states,mode)
        return en_output,output

