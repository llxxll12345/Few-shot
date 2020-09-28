import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

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


def get_state_dict(model):
    dicts = {}
    for key, val in model.state_dict().items():
        dicts[key] = val.clone()
    return dicts


class ConvModel(nn.Module):
    def __init__(self, in_dim=3, hid_dim=64, out_dim=64, img_sz=64, n_class=0):
        super().__init__()
        #self.attn = Attention(in_dim, img_size)
        self.conv1 = conv_block(in_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, hid_dim)
        self.conv3 = conv_block(hid_dim, hid_dim)
        self.conv4 = conv_block(hid_dim, out_dim)
        self.n_class = n_class
        self.reduce_dim = int(2 ** 4)
        if n_class != 0: 
            img_dim = img_sz // self.reduce_dim
            self.linear = nn.Linear(img_dim * img_dim * out_dim, n_class)

    def forward(self, x, state=None):
        if state != None:
            self.load_state_dict(state)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv2(x)
        print(x.view(x.size(0), -1).size())
        if self.n_class != 0:
            output = self.linear(x.view(x.size(0), -1))
            print(output.shape)
            return F.log_softmax(output, dim=1)
        return x.view(x.size(0), -1)

def test():
    model = ConvModel(n_class=5)
    x = torch.randn((1, 3, 64, 64))
    y = model(x)
    print(y.shape)
    dc =get_state_dict(model)
    print(model.state_dict()['conv1.0.bias'])
    for d in dc:
        print(d)
    #ps = model.parameters()
    #for p in ps:
    #    print(p)

test()
