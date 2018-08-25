import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as Variable

class transformNet(nn.Module):

    '''
    transformNet Model: mlp(64, 128, 1024) --> maxpool --> fc(512, 256) --> matrix multiplication (input * transform)
    '''

    def __init__(self, num_point=2000, K=3):
        super(transformNet, self).__init__()

        #input size
        self.N = num_point

        #num of dimension
        self.K = K

        # Initialize identity matrix on the GPU (do this here so it only happens once)
        # self.identity = Variable.Variable(torch.eye(self.K).double().view(-1).cuda())
        self.identity = Variable.Variable(torch.eye(self.K).float().view(-1))

        #Input: K@(N*1), Output: 64@(N*1)
        self.tNet1 = nn.Sequential(
            nn.Conv1d(self.K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Input: 64@(N*1), Output: 128@(N*1)
        self.tNet2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Input: 128@(N*1), Output: 1024@(N*1)
        self.tNet3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Input: 1024, Output: K*K
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, K*K)
        )

    def forward(self, x):

        #T-Net convolution
        #Input: B@K@N, Output: B@1024@N
        out = self.tNet1(x)
        out = self.tNet2(out)
        out = self.tNet3(out)

        #T-net maxpool
        #Input: B@1024@N, Output: B@1024@1 --> B@1024
        out = F.max_pool1d(out, self.N).squeeze(2)

        #Input: B@1024, Output: B@(K*K)
        out = self.fc(out)

        # Add identity matrix to transform
        # Output is still B x K^2 (broadcasting takes care of batch dimension)
        out += self.identity

        # Reshape the output into B x K x K affine transformation matrices
        out = out.view(-1, self.K, self.K)


        return out