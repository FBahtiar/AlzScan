import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import math
import os
import gc
import numpy as np
from tqdm import tqdm
from itertools import cycle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing

__all__ = ['effnetv2_l', 'effnetv2_xl', 'ClassificationBertSelfAttention', 'IntegratedModel']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        return x + self.conv(x) if self.identity else self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.0):
        super().__init__()
        self.cfgs = cfgs
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = MBConv

        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

def effnetv2_l(**kwargs):
    cfgs = [
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


class ClassificationBertSelfAttention(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_of_attention_heads"]
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        context = torch.matmul(attn_probs, v)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.all_head_size)
        output = self.dense(context)
        logits = self.classifier(output[:, 0, :])
        return logits


class IntegratedModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.efficientnet = effnetv2_l()
        self.efficientnet.classifier = nn.Identity()

        self.pre_attention_layers = nn.Sequential(
            nn.LayerNorm(1792),
            nn.Linear(1792, config["hidden_size"]),
            nn.Dropout(config.get("dropout_rate", 0.3)),
            nn.GELU()
        )

        self.mhsa = ClassificationBertSelfAttention(config, num_classes)

    def forward(self, x):
        features = self.efficientnet.features(x)
        features = self.efficientnet.conv(features)
        b, c, h, w = features.shape
        features = features.view(b, h * w, c)
        features = self.pre_attention_layers(features)
        logits = self.mhsa(features)
        return logits
