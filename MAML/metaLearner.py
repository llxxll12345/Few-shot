import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from MAML.model import *
from copy import deepcopy

class MetaLearner(nn.Module):
    def __init__(self, in_channels, n_class, args):
        super().__init__()
        self.n_way = args.n_way
        self.shot = args.shot
        self.query = args.query

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        
        self.n_update = args.n_update
        self.n_update_test = args.n_update_test

        self.conv_model = ConvModel(in_channels, n_class=n_class, img_sz=args.img_size)
        self.meta_optim = optim.Adam(self.conv_model.parameters(), lr=self.meta_lr)
    
    # forward with support and query (multi-task)
    def forward(self, sx, sy, qx, qy):
        n_task, n_support, _, h, w = sx.size()
        n_query = qx.size(1)

        total_q_loss = 0
        total_match = 0

        for i in range(n_task):
            out = self.conv_model(sx[i])
            loss = F.cross_entropy(out, sy[i])
            grad = torch.autograd.grad(loss, self.conv_model.parameters())

            st_dict = get_state_dict(self.conv_model)
            cur_dict = dict(map(lambda k: {k[0][0], k[0][1] - self.update_lr * k[1]}, zip(st_dict, grad)))
            # First two update steps are no grad for query
            for j in range(2):
                with torch.no_grad():
                    out = self.conv_model(qx[i], self.conv_model.state_dict())
                    loss = F.cross_entropy(out, qy[i])
                    total_q_loss += loss
                    prediction = F.softmax(out, dim=1).argmax(dim=1)
                    match = torch.eq(prediction, qy[i]).sum().item()
                    total_match += match

            # For the other update steps, first support then query
            for k in range(2, self.n_update + 1):
                out = self.conv_model(sx[i], cur_dict)
                loss = F.cross_entropy(out, sy[i])
                grad = torch.autograd.grad(loss, cur_dict)
                cur_dict = dict(map(lambda k: {k[0][0], k[0][1] - self.update_lr * k[1]}, zip(cur_dict, grad)))

                out_q = self.conv_model(qx[i], cur_dict)
                loss_q = F.cross_entropy(out_q, qy[i])
                total_q_loss += loss_q
                with torch.no_grad():
                    prediction = F.softmax(out_q, dim=1).argmax(dim=1)
                    match = torch.eq(prediction, qy[i]).sum().item()
                    total_match += match
        
        q_loss = total_q_loss/n_task
        self.meta_optim.zero_grad()
        q_loss.backward()
        self.meta_optim.step()
        accuracy = np.array(total_match) / (n_query * n_task)

        return q_loss, accuracy

    # testing (single task)
    def single_task(self, sx, sy, qx, qy):
        n_query = qx.size(0)
        total_match = 0
        total_loss = 0

        conv_model = deepcopy(self.conv_model)
        out = conv_model(sx)
        loss = F.cross_entropy(out, sy)
        grad = torch.autograd.grad(loss, conv_model.parameters())
        st_dict = get_state_dict(self.conv_model)
        cur_dict = dict(map(lambda k: {k[0][0], k[0][1] - self.update_lr * k[1]}, zip(st_dict, grad)))

        for j in range(2):
            with torch.no_grad():
                out = self.conv_model(qx, self.conv_model.state_dict())
                loss = F.cross_entropy(out, qy)
                prediction = F.softmax(out, dim=1).argmax(dim=1)
                match = torch.eq(prediction, qy).sum().item()
                total_match += match
                total_loss += loss

        for k in range(2, self.n_update_test + 1):
            out = self.conv_model(sx, cur_dict)
            loss = F.cross_entropy(out, sy)
            grad = torch.autograd.grad(loss, cur_dict)
            cur_dict = dict(map(lambda k: {k[0][0], k[0][1] - self.update_lr * k[1]}, zip(cur_dict, grad)))

            out_q = self.conv_model(qx, cur_dict)
            loss_q = F.cross_entropy(out_q, qy)
            total_loss += loss_q
            with torch.no_grad():
                prediction = F.softmax(out_q, dim=1).argmax(dim=1)
                match = torch.eq(prediction, qy).sum().item()
                total_match += match

        accuracy = total_match / (n_query * self.n_update_test)
        return total_loss, accuracy


def test():
    model = ConvModel(n_class=5)
    x = torch.randn((1, 3, 64, 64))
    y = model(x)
    print(y.shape)
    #m = MetaLearner(64, 5)
    #print(m.get_state_dict())
    #dc = m.get_state_dict()
    #for d in dc:
    #    print(d) d

test()