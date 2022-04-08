import torch
from torch import nn
import torchvision.models as models


class DeepModel(nn.Module):

    def __init__(self, gls_create: function):
        super(DeepModel, self).__init__()
        self.flatten = nn.Flatten()
        self.gls_create = gls_create


    def forward(self, x: torch.Tensor, logits: bool=False):
        x = self.base(x)
        x = self.flatten(x)
        x = self.gls(x, return_logits=logits)

        return x


    def train(self, x: torch.Tensor, y: torch.Tensor):
        x = self.base(x)
        x = self.flatten(x)
        self.gls.train(x, y)


class VGG(DeepModel):

    def __init__(self, gls_create: function, final_size=(1, 1)):
        super(VGG, self).__init__(gls_create)
        self.vgg = models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features[:-1] # Remove last pooling layer to fix sizing on mnist
        self.vgg_avg = nn.AdaptiveAvgPool2d(output_size=final_size)
        self.base = nn.Sequential(self.vgg_features, self.vgg_avg)
        self.gls = self.gls_create(512)


class ResNet(DeepModel):

    def __init__(self, gls_create: function):
        super(ResNet, self).__init__(gls_create)
        self.resnet = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(self.resnet.children())[:-1]) # remove last layer
        self.gls = self.gls_create(2048)

class EfficientNet(DeepModel):

    def __init__(self, gls_create: function):
        super(EfficientNet, self).__init__(gls_create)
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        self.base = nn.Sequential(*list(self.efficientnet.children())[:-1]) # remove last layer
        self.gls = self.gls_crate(2560)
