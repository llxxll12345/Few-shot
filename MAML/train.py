from sampler import Sampler

import torch
import numpy as np
import os
import argparse
import datetime

from metaLearner import MetaLearner
from model import ConvModel, get_state_dict
from torch.utils.data import DataLoader
from omniglot_d import Omniglot
from sampler import Sampler
from utils import *

def get_trainable_tensors(model):
    p = filter(lambda x: x.requires_grad, model.parameters())
    return sum(map(lambda x: np.prod(x.shape), p))


def save_model(model, name, save_path):
    torch.save(model.state_dict(), os.path.join(save_path, name+'.pth'))


def load_model(model, name, save_path):
    path = os.path.join(save_path, name + '.pth')
    if os.path.exists(os.path.join(save_path, name + '.pth')):
        print('load from previous model')
        pre_model = torch.load(path)
        model.load_state_dict(pre_model,strict=False)
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    renew_path(args.save)
    
    batch_size = args.shot+args.query
    train_set = Omniglot("dataset", 'train', download=True)
    train_sampler = Sampler(train_set.labels, args.n_batch_train, args.n_way, batch_size, limit_class=args.limit_class)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    test_set = Omniglot("dataset", 'test', download=True)
    test_sampler = Sampler(test_set.labels, args.n_batch_test, args.n_way, batch_size, limit_class=args.limit_class)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    maml = MetaLearner(in_channels=3, n_class=args.n_way, args=args).to(device)
    maml = load_model(maml, 'maml', args.save)

    training_log = {}
    training_log['args'] = args
    training_log['train_loss'] = []
    training_log['val_loss'] = []
    training_log['train_acc'] = []
    training_log['val_acc'] = []

    for epoch in range(1, args.epoch + 1):
        time_a = datetime.datetime.now()
        loss, accuracy = maml.train_whole_batch(train_loader)

        #if epoch % 10 == 0:
        print('Epoch {} training accuracy: {} loss: {}'.format(epoch, accuracy, loss))
        
        accuracy_list = []
        total_loss = 0
        
        for i, batch in enumerate(test_loader, 1):
            num = args.shot * args.n_way
            support_x, query_x = batch[0][:num].to(device), batch[0][num:].to(device)
            #support_y, query_y = batch_content[1][:num].to(device), batch_content[1][num:].to(device)
            support_y = torch.arange(args.n_way).repeat(args.shot)
            query_y = torch.arange(args.n_way).repeat(args.query)

            support_y = support_y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
            query_y   = query_y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

            # split to single task each time
            
            test_loss, test_accuracy = maml(support_x, support_y, query_x, query_y)
            accuracy_list.append(test_accuracy)
            total_loss += test_loss

        acc = np.array(accuracy_list).mean(axis=0).astype(np.float16)
        tloss = total_loss / len(test_loader)
        print('Validation accuracy: {} loss: {}'.format(acc, tloss))
                
        training_log['train_loss'].append(loss)
        training_log['train_acc'].append(accuracy)
        training_log['val_loss'].append(acc)
        training_log['val_acc'].append(tloss)

        torch.save(training_log, os.path.join(args.save, 'training_log'))
        
        if epoch % 100 == 0:
            save_model(maml, 'model', args.save)
        
        time_b = datetime.datetime.now()
        print('ETA:{}s/{}s'.format((time_b - time_a).seconds, (time_b - time_a).seconds * (args.epoch - epoch)))

    torch.save(training_log, os.path.join(args.save, 'training_log'))
    save_model(maml, 'model', args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',  '--epoch', type=int, default=200)
    # meta-batch size
    parser.add_argument('-b',  '--n_batch_train', type=int, default=8)
    parser.add_argument('-bt', '--n_batch_test', type=int, default=8)
    parser.add_argument('-w',  '--n_way', type=int, default=5)

    parser.add_argument('-s',  '--shot', type=int, default=1)
    parser.add_argument('-q',  '--query', type=int, default=3)

    parser.add_argument('-ml', '--meta_lr', type=int, default=1e-3)
    parser.add_argument('-ul', '--update_lr', type=int, default=0.4)
    parser.add_argument('-u',  '--n_update', type=int, default=2)
    parser.add_argument('-ut', '--n_update_test', type=int, default=2)
    parser.add_argument('-is', '--img_size', type=int, default=84)
    parser.add_argument('-l',  '--limit_class', type=bool, default=False)
    parser.add_argument('-sv', '--save', default='./model/proto')
    args = parser.parse_args()
    print(vars(args))
    train(args)

