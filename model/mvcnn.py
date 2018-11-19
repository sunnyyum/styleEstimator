import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self.make_layers()

    def make_layers(self):
        cfg = [32,'M', 64, 'M', 128, 'M', 256, 'M', 'L']
        layers = []
        in_channels = 3
        drop_rate = 0.1
        for index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.BatchNorm2d(cfg[index-1]),
                           nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(p=drop_rate)]
                drop_rate = drop_rate + 0.1

            if x == 'L':
                layers += [nn.Conv2d(in_channels, 512, kernel_size=1),
                           nn.ELU(inplace=True)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ELU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = x.transpose(0, 1)

        view_pool = []

        for v in x:
            v = self.features(v)
            v = v.view(v.size()[0],-1)

            view_pool.append(v)

        # pooled_view = view_pool[0]
        #
        # for i in range(1, len(view_pool)):
        #     pooled_view = torch.max(pooled_view, view_pool[i])

        return view_pool

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative


def mvcnn(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model