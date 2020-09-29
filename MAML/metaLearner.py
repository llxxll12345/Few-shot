import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from model import ConvModel, get_state_dict
from copy import deepcopy

class MetaLearner(nn.Module):
    def __init__(self, in_channels, n_class, args):
        super().__init__()
        self.n_way = args.n_way
        self.shot = args.shot
        self.query = args.query
        self.n_class = n_class

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        
        self.n_update = args.n_update
        self.n_update_test = args.n_update_test

        self.conv_model = ConvModel(in_channels, n_class=self.n_class, img_sz=args.img_size)
        self.meta_optim = optim.Adam(self.conv_model.parameters(), lr=self.meta_lr)

    def train_whole_batch(self, train_loader):
        total_loss = 0
        total_match = 0

        n_test = len(train_loader)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        for i, batch in enumerate(train_loader, 1):
            num = self.shot * self.n_class
            support_x, query_x = batch[0][:num].to(device), batch[0][num:].to(device)
            #support_y, query_y = batch[1][:num].to(device), batch[1][num:].to(device)
            support_y = torch.arange(self.n_class).repeat(self.shot)
            query_y = torch.arange(self.n_class).repeat(self.query)

            support_y = support_y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
            query_y   = query_y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

            loss, matches = self.forward(support_x, support_y, query_x, query_y)
            total_loss += loss
            total_match += matches
        
        total_loss /= n_test
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()
        accuracy = total_match / (n_test * self.query)
        return total_loss, accuracy

    def update_weights(self, grad):
        cur_dict = deepcopy(self.conv_model.state_dict())
        i = 0
        for name, param in self.conv_model.named_parameters():
            cur_dict[name] = param - self.update_lr * grad[i]
            i += 1
        return cur_dict

    # forward with support and query (single task)
    def forward(self, sx, sy, qx, qy):
        n_query = qx.size(0)
        total_match = 0
        total_loss = 0

        out = self.conv_model(sx)
        loss = F.cross_entropy(out, sy)
        grad = torch.autograd.grad(loss, self.conv_model.parameters())
        cur_dict = self.update_weights(grad)

        '''
        for j in range(2):
            with torch.no_grad():
                out = self.conv_model(qx)
                loss = F.cross_entropy(out, qy)
                prediction = F.softmax(out, dim=1).argmax(dim=1)

                print(loss.shape, prediction.shape)
                match = torch.eq(prediction, qy).sum().item()
                total_match += match
                total_loss += loss
        '''

        for k in range(self.n_update_test + 1):
            self.conv_model.load_state_dict(cur_dict)
            out = self.conv_model(sx)
            loss = F.cross_entropy(out, sy)
            grad = torch.autograd.grad(loss, self.conv_model.parameters())
            cur_dict = self.update_weights(grad)

            self.conv_model.load_state_dict(cur_dict)
            out_q = self.conv_model(qx)
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
