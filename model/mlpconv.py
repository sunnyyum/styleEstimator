import torch.nn as nn
import torch.nn.functional as F

class mlpConv(nn.Module):
    '''
    mlpConv net: conv(n*n*n) --> relu --> conv(1*1*1) --> relu --> conv(1*1*1)
    '''

    def __init__(self, in_channels, n_classes):
        super(mlpConv, self).__init__()

        #Input: 1@(30*30*30), Output: 48@(13*13*13)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=48, kernel_size=6, stride=2), #output shape(48,13,13,13)
            nn.ReLU(),
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=1, stride=1), #output shape(48,13,13,13)
            nn.ReLU(),
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=1, stride=1), #output shape(48,13,13,13)
        )

        #Input: 48@(13*13*13), Output: 160@(5*5*5)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=160, kernel_size=5, stride=2), #output shape(160,5,5,5)
            nn.ReLU(),
            nn.Conv3d(in_channels=160, out_channels=160, kernel_size=1, stride=1), #output shape(160,5,5,5)
            nn.ReLU(),
            nn.Conv3d(in_channels=160, out_channels=160, kernel_size=1, stride=1), #output shape(160,5,5,5)
        )

        #Input: 160@(5*5*5), Output: 512@(2*2*2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=160, out_channels=512, kernel_size=3, stride=2), #output shape(512,2,2,2)
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=1, stride=1), #output shape(512,2,2,2)
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=1, stride=1), #output shape(512,2,2,2)
        )

        #fully connected layer
        #fc --> relu --> fc --> relu --> fc
        #Input: (512@(2*2*2))=4096, Output: 256
        self.fc = nn.Sequential(
            nn.Linear(512*2*2*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    #forward for positive and negative examples
    def forward_once(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out

    #concatenate features and feed to fully conncected layers
    def forward(self, positive, negative):
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        out_p = output_positive.view(output_positive.size()[0], -1)
        out_n = output_negative.view(output_negative.size()[0], -1)

        out_fc_p = self.fc(out_p)
        out_fc_n = self.fc(out_n)

        return out_fc_p, out_fc_n



