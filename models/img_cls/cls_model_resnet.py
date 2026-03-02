import torch
import torch.nn as nn
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import torch
import torch.nn as nn

class ImageClsBackbone(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        self.conv_reduce = nn.Conv2d(512, 256, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, dict_item):
        images = dict_item['cam_front_img'].cuda()

        feats = self.feature_extractor(images)
        feats = self.conv_reduce(feats)
        feats = self.gap(feats).view(feats.size(0), -1)

        output = self.fc(feats)

        dict_item['img_cls_output'] = output
        return dict_item
