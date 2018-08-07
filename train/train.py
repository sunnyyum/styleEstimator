import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import mlpconv
from model import triplet
from supplementary import dataset



#hyper paramerters
lr = 0.001
num_epoch = 10
num_classes = 3

#device setup
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

#test value
# Create random Tensors to hold inputs, outputs, target
# input_anchor = torch.randn(1,1,30,30,30, device=device)
# input_positive = torch.randn(1,1,30,30,30, device=device)
# input_negative = torch.randn(1,1,30,30,30, device=device)

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

print("Training..\n")

#parameters for total loss value
total_loss = 0
num_sample = len(style_train_loader)

#start training
for epoch in range(0, num_epoch):
    loss = []
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

        total_loss += loss_output.item()
        if i % 100 == 0:
            print("epoch: {}\nloss: {:0.2f}\n".format(epoch, loss_output))
            loss.append(loss_output.item())

print("End\n")

total_loss /= num_sample
print("total loss: {:0.2f}\n".format(total_loss))









