import torch.nn as nn
import torchvision.models as models
import os, ssl

# set up for downloading pretrained model
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

class VGG(nn.Module):
    '''
    VGG16 Model
    '''
    def __init__(self, num_classes=1000, pretrained=False):
        super(VGG, self).__init__()

        self.features=self.make_layers(pretrained=pretrained)

    def make_layers(self, pretrained=False):

        model = models.vgg16_bn(pretrained=pretrained)
        layer_list = list(model.children())[:-1]

        return nn.Sequential(*layer_list[0][0:24])

    def forward_once(self, x):
        x = x.transpose(0, 1)

        view_pool = []

        for v in x:
            v = self.features(v)
            v = v.view(v.size()[0],-1)

            view_pool.append(v)

        return view_pool

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative


class ResNet(nn.Module):
    '''
    resnet18 Model
    '''
    def __init__(self, pretrained=False):
        super(ResNet, self).__init__()

        self.features = self.make_layers(pretrained=pretrained)

    def make_layers(self, pretrained=False):
        model = models.resnet18(pretrained=pretrained)

        return nn.Sequential(*list(model.children())[:-2])

    def forward_once(self, x):
        x = x.transpose(0, 1)

        view_pool = []

        for v in x:
            v = self.features(v)
            v = v.view(v.size()[0],-1)

            view_pool.append(v)

        return view_pool

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative

