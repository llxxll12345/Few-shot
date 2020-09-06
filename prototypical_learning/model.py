import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
# Very basic convolution model

def conv_block(in_dim, out_dim):
    batch_norm = nn.BatchNorm2d(out_dim)
    # initialize batch norm layer
    nn.init.uniform_(batch_norm.weight)
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, padding=1),
        batch_norm,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ShortCutBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(out_dim)
        # initialize batch norm layer
        nn.init.uniform_(self.batch_norm.weight)
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_0 = x
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x += x_0
        return x

class Attention(nn.Module):
    def __init__(self, in_dim, img_size):
        super().__init__()
        # Linear model to cuda should be specially specified!!!
        sz = img_size * img_size * in_dim
        self.attn = nn.Linear(sz, sz)

    def forward(self, x):
        flat_x = x
        flat_x = flat_x.view(x.shape[0], -1)
        attn_weight = F.softmax(self.attn(flat_x), dim=0).view(x.shape)
        x =torch.matmul(x, attn_weight)
        return x

class ConvModel(nn.Module):
    def __init__(self, in_dim=3, hid_dim=64, out_dim=64, img_size=64):
        super().__init__()
        #self.attn = Attention(in_dim, img_size)
        self.conv1 = conv_block(in_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, hid_dim)
        self.conv3 = conv_block(hid_dim, hid_dim)
        self.conv4 = conv_block(hid_dim, out_dim)

    def forward(self, x):
        #x = self.attn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv2(x)
        return x.view(x.size(0), -1)
