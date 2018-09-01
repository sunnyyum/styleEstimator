import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as Variable

from model import transformNet

class pointNet(nn.Module):

    '''
    pointNet Classification model: input transformation --> mlp(64,64) --> feature transformation --> mlp(64,128,1024) --> maxpool
    In this code, it excludes the adaptive batch normalization decay rate
    '''

    def __init__(self, num_points = 2000, K=3, num_classes = 40):
        super(pointNet, self).__init__()

        #input transformation
        self.input_transform = transformNet.transformNet(num_point=num_points, K=K)

        #feature transformation
        self.feature_transform = transformNet.transformNet(num_point=num_points, K=64)

        #mlp(64,64)
        #Input: B@K@N, Output: B@64@N
        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        #mlp(64,128,1024)
        #Input: B@64@N, Output: B@1024@N
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )


    # forward for examples
    def forward_once(self, x):
        # Number of points put into the network
        N = x.shape[2]

        # First compute the input data transform and transform the data
        # T1 is B x K x K and x is B x K x N, so output is B x K x N
        T1 = self.input_transform(x)
        out = torch.bmm(T1, x)

        # Run the transformed inputs through the first feature MLP
        # Output is B x 64 x N
        out = self.mlp1(out)

        # Transform the embeddings. This gives us the "local feature"
        # referred to in the paper/slides
        # T2 is B x 64 x 64 and x is B x 64 x N, so output is B x 64 x N
        T2 = self.feature_transform(out)
        local_feature = torch.bmm(T2, out)

        # Further embed the "local feature"
        # Output is B x 1024 x N
        global_feature = self.mlp2(local_feature)

        # Pool over the number of points. This results in the "global feature"
        # referred to in the paper/slides
        # Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
        global_feature = F.max_pool1d(global_feature, N).squeeze(2)

        return global_feature, local_feature, T2

    # concatenate features and feed to fully conncected layers
    def forward(self, anchor, positive, negative):

        output_anchor, _, T2A = self.forward_once(anchor)
        output_positive, _, T2P = self.forward_once(positive)
        output_negative, _, T2N = self.forward_once(negative)

        return output_anchor, output_positive, output_negative