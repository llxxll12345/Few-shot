from MAML.sampler import Sampler

import torch
import numpy as np
import os
import argparse
import datetime

from MAML.metaLearner import *
from MAML.model import *
from torch.utils.data import DataLoader
from MAML.omniglot_d import Omniglot
from MAML.sampler import Sampler
from MAML.utils import *

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


train_folder = 'dataset/images_background'
test_folder = 'dataset/images_evaluation'


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    renew_path(args.save)
    
    batch_size = args.shot+args.query
    train_set = Omniglot(train_folder, 'train')
    train_sampler = Sampler(train_set.label, args.n_batch_train, args.train_way, batch_size, limit_class=args.limit_class)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    test_set = Omniglot(test_folder, 'test')
    test_sampler = Sampler(test_set.label, args.n_batch_test, args.test_way, batch_size, limit_class=args.limit_class)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    maml = MetaLearner(in_channels=3, n_class=args.train_way, args=args).to(device)
    maml = load_model(maml, 'maml', args.save)

    training_log = {}
    training_log['args'] = args
    training_log['train_loss'] = []
    training_log['val_loss'] = []
    training_log['train_acc'] = []
    training_log['val_acc'] = []

    for epoch in range(1, args.epoch + 1):
        time_a = datetime.datetime.now()
        batch_content = next(train_loader)
        num = args.shot * args.train_way
        support_x, query_x = batch_content[0][:num].to(device), batch_content[0][num:].to(device)
        support_y, query_y = batch_content[1][:num].to(device), batch_content[1][num:].to(device)
        loss, accuracy = maml(support_x, support_y, query_x, query_y)

        if epoch % 10 == 0:
            print('Epoch {} training accuracy: {} loss: {}'.format(epoch, accuracy, loss))

        
        accuracy_list = []
        loss_list = []
        for _ in range(1000//args.task_num):
            batch_content = next(test_loader)
            num = args.shot * args.test_way
            support_x, query_x = batch_content[0][:num].to(device), batch_content[0][num:].to(device)
            support_y, query_y = batch_content[1][:num].to(device), batch_content[1][num:].to(device)
            
            # split to single task each time
            for sx, sy, qx, qy in zip(support_x, support_y, query_x, query_y):
                test_loss, test_accuracy = maml.finetunning(sx, sy, qx, qy)
                accuracy_list.append(test_accuracy)
                loss_list.append(test_loss)

        acc = np.array(accuracy_list).mean(axis=0).astype(np.float16)
        tloss = np.array(loss_list).mean(axis=0).astype(np.float16)
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
    parser.add_argument('-b',  '--n_batch_train', type=int, default=16)
    parser.add_argument('-bt', '--n_batch_test', type=int, default=16)
    parser.add_argument('-s',  '--shot', type=int, default=1)
    parser.add_argument('-q',  '--query', type=int, default=15)
    parser.add_argument('-ml', '--meta_lr', type=int, default=0.01)
    parser.add_argument('-ul', '--update_lr', type=int, default=0.01)
    parser.add_argument('-u',  '--n_update', type=int, default=10)
    parser.add_argument('-ut', '--n_update_test', type=int, default=10)
    parser.add_argument('-is', '--img_size', type=int, default=84)
    parser.add_argument('-l',  '--limit_class', type=bool, default=False)
    parser.add_argument('-sv', '--save', default='./model/proto')
    args = parser.parse_args()
    print(vars(args))
    train(args)

