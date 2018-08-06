import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np

from model import mlpconv
from model import triplet
from supplementary import dataset


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
input_anchor = torch.randn(1,1,30,30,30, device=device)
input_positive = torch.randn(1,1,30,30,30, device=device)
input_negative = torch.randn(1,1,30,30,30, device=device)



num_classes = 3

#put to dataloader
#this part needs to add with 3d Data
style_folder = Dataset()
style_dataset = dataset.StyleEstimatorDataset(style_folder)
style_train_loader = DataLoader(style_dataset, batch_size=1)

#set up model
model = mlpconv.mlpConv(in_channels=1,n_classes=num_classes)
criterion = triplet.TripletLoss()

#set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training..")

for epoch in range(0, num_epoch):
    # it should be changed with train funciton
    loss = []
    total_loss = 0

    model.train()

    for i, (input_anchor, input_positive, input_negative) in enumerate(style_train_loader):
    # for i in range(10):

        if cuda:
            input_anchor, input_positive, input_negative = input_anchor.cuda(), input_positive.cuda(), input_negative.cuda()
        optimizer.zero_grad()
        output_anchor, output_positive, output_negative = model(input_anchor, input_positive, input_negative)

        loss_output = criterion(Variable(output_anchor.float()), Variable(output_positive.float()), Variable(output_negative.float()))
        loss_output = Variable(loss_output, requires_grad=True)
        # loss_output = criterion(output_anchor, output_positive, output_negative)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output
        if i % 100 == 0:
            print("epoch: {}\n loss: {}\n".format(epoch, loss_output))
            loss.append(loss_output)

print("End\n")







