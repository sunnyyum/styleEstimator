import torch
import torch.utils.data as Data

import numpy as np

from model import mlpconv
from model import triplet


def train(train_loader, model, criterion, optimizer, cuda):

    loss = []
    total_loss = 0


    for i,(input,target) in enumerate(train_loader):
        if cuda:
            input, target = input.cuda(), target.cuda()

        input_positive, input_negative = input

        optimizer.zero_grad()
        output_positive, output_negative = model(input_positive, input_negative)

        loss_output = criterion(target, output_positive, output_negative)
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output
        if i % 100 == 0:
            print("epoch: {}\n loss: {}\n".format(i,loss_output))
            loss.append(loss_output)

    total_loss /= (i+1)
    return total_loss

#hyper paramerters
lr = 0.001

#device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#test value
# Create random Tensors to hold inputs and outputs
input_positive = torch.randn(30,30,30, device=device)
output_positive = 1

input_negative = torch.randn(30,30,30, device=device)
output_negative = 2

num_classes = 3

#put to dataloader
train_loader_positive = Data.DataLoader(dataset=input_positive)

#set up model
model = mlpconv(in_channels=1,n_classes=num_classes)
criterion = triplet(margin=0.01)

#set up optimizer
optimizer = torch.optim.Adam(model.parameter(), lr=lr)






