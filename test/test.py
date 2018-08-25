import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import pointNet
from model import triplet

from supplementary import dataset

#hyper paramerters
num_epoch = 10
num_classes = 40
batch_size = 2

#device setup
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

#test value
# Create random Tensors to hold inputs, outputs, target
# input_anchor = torch.randn(2, 3, 2000, device=device)
# input_positive = torch.randn(2, 3, 2000, device=device)
# input_negative = torch.randn(2, 3, 2000, device=device)

#put to dataloader
#this part needs to add with 3d Data
style_folder = Dataset()
style_dataset = dataset.PointNetStyleEstimator(style_folder)
style_train_loader = DataLoader(style_dataset, batch_size=batch_size)

#set up model
model = pointNet.pointNet(num_points=2000, K=3, num_classes=num_classes)
criterion = triplet.TripletLoss()



print("Testing..\n")


loss = []
total_loss = 0

model.eval()
for i,(input_anchor, input_positive, input_negative) in enumerate(style_train_loader):
# for i in range(1):
    if cuda:
        input_anchor, input_positive, input_negative = input_anchor.cuda(), input_positive.cuda(), input_negative.cuda()


    output_anchor, output_positive, output_negative = model(input_anchor, input_positive, input_negative)

    loss_output = criterion(Variable(output_anchor.float()), Variable(output_positive.float()),
                            Variable(output_negative.float()))

    print("Test #{}:\nloss: {:0.2f}\n".format(i, loss_output))

    #save the loss output for each test sample
    loss_data = (i, loss_output.item())

    loss.append(loss_data)

print("End\n")




