import torch
from torch import nn
import torchvision.models as models

from gls import GLSLayer


class DeepModel(nn.Module):

    def __init__(self, gls: GLSLayer):
        super(DeepModel, self).__init__()
        self.flatten = nn.Flatten()
        self.gls = gls


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

    def __init__(self, gls: GLSLayer, final_size=(7, 7)):
        super(VGG, self).__init__(gls)
        self.vgg = models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        self.vgg_avg = nn.AdaptiveAvgPool2d(output_size=final_size)
        self.base = nn.Sequential(self.vgg_features, self.vgg_avg)


class ResNet(nn.Module):

    def __init__(self, gls: GLSLayer):
        super(ResNet, self).__init__(gls)
        self.resnet = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(self.resnet.children())[:-1]) # remove last layer


class EfficientNet(nn.Module):

    def __init__(self, gls: GLSLayer):
        super(EfficientNet, self).__init__(gls)
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        self.base = nn.Sequential(*list(self.efficientnet.children())[:-1]) # remove last layer
