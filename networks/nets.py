from networks.deeplabv3 import ResNetDeepLabv3

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
from bn import InPlaceABNSync


class AnchorDiffNet(nn.Module):
    def __init__(self, backbone='ResNet50', pyramid_pooling='deeplabv3', embedding=128, batch_mode='sync'):
        super(AnchorDiffNet, self).__init__()
        if pyramid_pooling == 'deeplabv3':
            self.features = ResNetDeepLabv3(backbone, num_classes=embedding, batch_mode=batch_mode)
        elif pyramid_pooling == 'pspnet':
            raise RuntimeError('Pooling module not implemented')
        else:
            raise RuntimeError('Unknown pyramid pooling module')
        self.cls = nn.Sequential(
            nn.Conv2d(3 * embedding, embedding, kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(embedding),
            nn.Dropout2d(0.10),
            nn.Conv2d(embedding, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, reference, current):
        ref_features = self.features(reference)[0]
        curr_features = self.features(current)[0]
        batch, channel, h, w = curr_features.shape
        M = h * w
        ref_features = ref_features.view(batch, channel, M).permute(0, 2, 1)
        curr_features = curr_features.view(batch, channel, M)

        p_0 = torch.matmul(ref_features, curr_features)
        p_0 = F.softmax((channel ** -.5) * p_0, dim=-1)
        p_1 = torch.matmul(curr_features.permute(0, 2, 1), curr_features)
        p_1 = F.softmax((channel ** -.5) * p_1, dim=-1)
        feats_0 = torch.matmul(p_0, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        feats_1 = torch.matmul(p_1, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([feats_0, feats_1, curr_features], dim=1).view(batch, 3 * channel, h, w)
        pred = self.cls(x)

        return pred


class ConcatNet(nn.Module):
    def __init__(self, backbone='ResNet50', pyramid_pooling='deeplabv3', embedding=128, batch_mode='sync'):
        super(ConcatNet, self).__init__()
        if pyramid_pooling == 'deeplabv3':
            self.features = ResNetDeepLabv3(backbone, num_classes=embedding, batch_mode=batch_mode)
        elif pyramid_pooling == 'pspnet':
            raise RuntimeError('Pooling module not implemented')
        else:
            raise RuntimeError('Unknown pyramid pooling module')
        self.cls = nn.Sequential(
            nn.Conv2d(2*embedding, embedding, kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(embedding),
            nn.Dropout2d(0.10),
            nn.Conv2d(embedding, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, reference, current):
        ref_features = self.features(reference)[0]
        curr_features = self.features(current)[0]
        x = torch.cat([ref_features, curr_features], dim=1)
        pred = self.cls(x)
        
        return pred


class InterFrameNet(nn.Module):
    def __init__(self, backbone='ResNet50', pyramid_pooling='deeplabv3', embedding=128, batch_mode='sync'):
        super(InterFrameNet, self).__init__()
        if pyramid_pooling == 'deeplabv3':
            self.features = ResNetDeepLabv3(backbone, num_classes=embedding, batch_mode=batch_mode)
        elif pyramid_pooling == 'pspnet':
            raise RuntimeError('Pooling module not implemented')
        else:
            raise RuntimeError('Unknown pyramid pooling module')
        self.cls = nn.Sequential(
            nn.Conv2d(2 * embedding, embedding, kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(embedding),
            nn.Dropout2d(0.10),
            nn.Conv2d(embedding, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, reference, current):
        ref_features = self.features(reference)[0]
        curr_features = self.features(current)[0]
        batch, channel, h, w = curr_features.shape
        M = h * w
        ref_features = ref_features.view(batch, channel, M).permute(0, 2, 1)
        curr_features = curr_features.view(batch, channel, M)

        p_0 = torch.matmul(ref_features, curr_features)
        p_0 = F.softmax((channel ** -.5) * p_0, dim=-1)
        feats_0 = torch.matmul(p_0, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([feats_0, curr_features], dim=1).view(batch, 2 * channel, h, w)
        pred = self.cls(x)

        return pred


class IntraFrameNet(nn.Module):
    def __init__(self, backbone='ResNet50', pyramid_pooling='deeplabv3', embedding=128, batch_mode='sync'):
        super(IntraFrameNet, self).__init__()
        if pyramid_pooling == 'deeplabv3':
            self.features = ResNetDeepLabv3(backbone, num_classes=embedding, batch_mode=batch_mode)
        elif pyramid_pooling == 'pspnet':
            raise RuntimeError('Pooling module not implemented')
        else:
            raise RuntimeError('Unknown pyramid pooling module')
        self.cls = nn.Sequential(
            nn.Conv2d(2*embedding, embedding, kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(embedding),
            nn.Dropout2d(0.10),
            nn.Conv2d(embedding, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, current):
        curr_features = self.features(current)[0]
        batch, channel, h, w = curr_features.shape
        M = h * w
        curr_features = curr_features.view(batch, channel, M)
        
        p_1 = torch.matmul(curr_features.permute(0, 2, 1), curr_features)
        p_1 = F.softmax((channel**-.5) * p_1, dim=-1)
        feats_1 = torch.matmul(p_1, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([feats_1, curr_features], dim=1).view(batch, 2*channel, h, w)
        pred = self.cls(x)
        
        return pred
