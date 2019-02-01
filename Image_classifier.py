import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Image_classifier(nn.Module):
    def __init__(self,input_height,input_width,input_channels):
        super(Image_classifier,self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.conv_layer1 = nn.Conv2d(input_channels,13,(5,5*13),stride = (1,13),bias = True)
        self.max_pool1 = nn.MaxPool2d(6,2)
        self.conv_layer2 = nn.Conv2d(13,5,6,stride=2,bias=True)
        self.max_pool2 = nn.MaxPool2d(2,2)
        self.conv_layer3 = nn.Conv2d(5,5,3,stride=1,bias=True)
        self.max_pool3 = nn.MaxPool2d(2,2)
        self.fc_layer1 = nn.Linear(20,3,bias=True)
    
    def forward(self,input_img):
        input_img = self.batch_norm(input_img)
        output = self.conv_layer1(input_img)
        output = F.relu(output)
        output1 = self.max_pool1(output)
        output2 = self.conv_layer2(output1)
        output2  = F.relu(output2)
        output3 = self.max_pool2(output2)
        output4 = self.conv_layer3(output3)
        output4 = F.relu(output4)
        output5 = self.max_pool3(output4)
        output5 = output5.reshape(output5.size(0),-1)
        output6 = self.fc_layer1(output5)
        #output = F.sigmoid(output)
        return output,output1,output2,output4,output5

