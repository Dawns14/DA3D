import torch
import torch.nn as nn
import numpy as np

class ImageClsBackbone(nn.Module):
    def __init__(self, cfg=None):
        super(ImageClsBackbone, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.adjust = nn.Linear(220, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 7)

    def forward(self, dict_item):
        img = dict_item['cam_front_img'].cuda()
        b, c, h, w = img.size()
        x1 = self.bn1(self.conv1(img))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.bn3(self.conv3(x2))
        x3 = x3.view(b, 64, -1)
        x4 = self.adjust(x3)
        x5 = self.gap(x4)
        x6 = self.fc(x5.squeeze())
        dict_item['img_cls_output'] = x6
        dict_item['img_cls_gap'] = x5
        dict_item['img_cls_feat'] = x4
        dict_item['x1'] = x1
        dict_item['x2'] = x2
        dict_item['x3'] = x3

        return dict_item
