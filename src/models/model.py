
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init
import pyclustering
import torch.nn.functional as F
from hyptorch.nn import ToPoincare
import timm

OVERFLOW_MARGIN = 1e-8
import math
import torch
import numpy as np


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class MainModel(torch.nn.Module):
    def __init__(self, option):
        super(MainModel, self).__init__()
        self.option = option
        ######################VGG################################
        self.backbone = torchvision.models.vgg19(pretrained=True)
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:6])

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.hash_layer = nn.Sequential(nn.Linear(4096, 512), nn.Dropout(0.1),
                                        nn.ReLU(), nn.Linear(512, option.hash_bit))
        self.toPoincare = ToPoincare(c=option.hyper_c,
                                     ball_dim=option.hash_bit,
                                     riemannian=False,
                                     clip_r=option.clip_r)
        self.head = nn.Sequential(nn.Linear(self.option.hash_bit, self.option.hyper_dim), self.toPoincare)


    def forward(self, images, Train: bool):
        if Train:
            ################## VGG #####################
            aug_one_pre = self.backbone.features(images[0])
            aug_one_pre = aug_one_pre.view(aug_one_pre.size(0), -1)
            aug_one_pre = self.backbone.classifier(aug_one_pre)
            aug_two_pre = self.backbone.features(images[1])
            aug_two_pre = aug_two_pre.view(aug_two_pre.size(0), -1)
            aug_two_pre = self.backbone.classifier(aug_two_pre)

            origin_image = self.backbone.features(images[2])
            origin_image = origin_image.view(origin_image.size(0), -1)
            origin_image = self.backbone.classifier(origin_image)

            h_1 = torch.tanh(self.hash_layer(aug_one_pre))
            h_2 = torch.tanh(self.hash_layer(aug_two_pre))
            h_3 = torch.tanh(self.hash_layer(origin_image))
            # project to Poincare ball (hyperbolic space)
            if self.option.hyper_c == 0:
                p_1 = h_1
                p_2 = h_2
                p_3 = h_3
            else:
                p_1 = self.head(h_1)
                p_2 = self.head(h_2)
                p_3 = self.head(h_3)
            """
            p_3: feature of image without augmentation in the hyperbolic space, for tree construction
            p_1: feature of image with augmentation a in the hyperbolic space
            p_2: feature of image with augmentation b in the hyperbolic space
            h_1: hash codes of image with augmentation a 
            h_2: hash codes of image with augmentation b 
            Note that p_1 \\approx h_1 when c -> 0
            """

            return p_3, h_3, p_1, p_2, h_1, h_2
        else:
            with torch.no_grad():
                ################## VGG ##################
                images = self.backbone.features(images)
                images = images.view(images.size(0), -1)
                images = self.backbone.classifier(images)

                hash_code = torch.tanh(self.hash_layer(images))

                return hash_code

    def getParams(self):
        return [
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
            {'params': self.head.parameters(), 'lr': self.option.lr}
        ]


if __name__ == '__main__':
    hash_codes = torch.tensor([[-1., 1., -1., 1.], [-1., 1., 1., 1.], [1., -1., -1., 1.], [-1., 1., 1., 1.]])
