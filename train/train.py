import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import numpy as np

from model import mlpconv
from model import triplet


def train(train_loader, model, criterion, optimizer, cuda):

    loss = []
    total_loss = 0

    model.train()

    for i,(input,target) in enumerate(train_loader):
        if cuda:
            input, target = input.cuda(), target.cuda()

        input_positive, input_negative = input

        optimizer.zero_grad()
        output_positive, output_negative = model(input_positive, input_negative)

        loss_output = criterion(target, output_positive, output_negative)

        optimizer.zero_grad()
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
num_epoch = 10

#device setup
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

#test value
# Create random Tensors to hold inputs, outputs, target
# input_positive = np.random.rand(30, 30 ,30)
# input_negative = np.random.rand(30, 30 ,30)
input_positive = torch.randn(1,1,30,30,30, device=device)
input_negative = torch.randn(1,1,30,30,30, device=device)

# output_positive = np.array([1])
# output_negative = np.array([2])
target = torch.tensor([1], device=device)

num_classes = 3

#put to dataloader
# dataset = TensorDataset(input_positive, input_negative)
# train_loader_positive = DataLoader(, batch_size=1)

#set up model
model = mlpconv.mlpConv(in_channels=1,n_classes=num_classes)
criterion = triplet.TripletLoss()

#set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training..")

for num_epoch in range(0, num_epoch):
    # it should be changed with train funciton
    loss = []
    total_loss = 0

    model.train()

    for i in range(10):


        optimizer.zero_grad()
        output_positive, output_negative = model(input_positive, input_negative)

        loss_output = criterion(Variable(target.float()), Variable(output_positive.float()), Variable(output_negative.float()))
        loss_output = Variable(loss_output, requires_grad=True)
        # loss_output = criterion(target, output_positive, output_negative)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output
        print("epoch: {}\n loss: {}\n".format(i, loss_output))
        loss.append(loss_output)

print("End\n")







