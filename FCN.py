import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#https://neptune.ai/blog/image-segmentation
class JPU(nn.Module):
    def __init__(self, in_channels, width=256, norm_layer=None):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.dilation1 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation4 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        feat3, feat4, feat5 = feats

        feat5 = self.conv5(feat5)
        feat4 = self.conv4(feat4)
        feat3 = self.conv3(feat3)

        h, w = feat5.size(2), feat5.size(3)
        feat4 = F.interpolate(feat4, size=(h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(feat3, size=(h, w), mode='bilinear', align_corners=True)

        feat4 = feat4 + self.dilation1(feat5)
        feat3 = feat3 + self.dilation2(feat4)
        feat3 = feat3 + self.dilation3(feat3)

        out = self.dilation4(feat3)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class JPU(nn.Module):
    def __init__(self, in_channels, width=256, norm_layer=None):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.dilation1 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation4 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        feat3, feat4, feat5 = feats

        feat5 = self.conv5(feat5)
        feat4 = self.conv4(feat4)
        feat3 = self.conv3(feat3)

        h, w = feat5.size(2), feat5.size(3)
        feat4 = F.interpolate(feat4, size=(h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(feat3, size=(h, w), mode='bilinear', align_corners=True)

        feat4 = feat4 + self.dilation1(feat5)
        feat3 = feat3 + self.dilation2(feat4)
        feat3 = feat3 + self.dilation3(feat3)

        out = self.dilation4(feat3)
        return out





class FastFCN(nn.Module):
    def __init__(self, num_classes,width=256):
        super(FastFCN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.jpu = JPU(in_channels=[256,512, 1024, 2048],width=width, norm_layer=nn.BatchNorm2d)

        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        feat2 = x
        x = self.layer2(x)
        feat3 = x
        x = self.layer3(x)
        feat4 = x
        x = self.layer4(x)
        feat5 = x

        x = self.jpu([feat3, feat4, feat5])
        x = self.head(x)
        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=True)
        return x

