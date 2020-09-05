import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sampler import Sampler
from model import ConvModel
import torch.nn as nn
from utils import *
from omniglot import OmiglotSet
import datetime
import os

def save_model(model, name, save_path):
    torch.save(model.state_dict(), os.path.join(save_path, name+'.pth'))

def load_model(model, name, save_path):
    path = os.path.join(save_path, 'model.pth')
    if os.path.exists(os.path.join(save_path, 'model.pth')):
        pre_model = torch.load(path)
        model.load_state_dict(pre_model,strict=False)
    return model

def train(args):
    """
        Terminology: k-way n-shot, k classes, n shots per class
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    renew_path(args.save)
    
    shots = args.shot+args.query
    train_set = OmiglotSet('train')
    train_sampler = Sampler(train_set.label, args.batch_size_train, args.train_way, shots)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    test_set = OmiglotSet('test')
    test_sampler = Sampler(test_set.label, args.batch_size_test, args.test_way, shots)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    model = ConvModel(img_size=84).to(device)
    model = load_model(model, 'model', args.save)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # learing rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = F.cross_entropy
    
    # training log
    training_log = {}
    training_log['args'] = []
    training_log['train_loss'] = []
    training_log['val_loss'] = []
    training_log['train_acc'] = []
    training_log['val_acc'] = []
    training_log['max_acc'] = 0.0

    for epoch in range(1, args.epoch + 1):
        time_a = datetime.datetime.now()
        model.train()
        average_loss = 0
        average_accuracy = 0
        for i, batch in enumerate(train_loader, 1):
            num = args.shot * args.train_way
            support_x, query_x = batch[0][:num].to(device), batch[0][num:].to(device)
            support_y, query_y = batch[1][:num].to(device), batch[1][num:].to(device)
            #print(support_x.shape)
            embedding = model(support_x)

            # Get the mean of all the embeddings to get the prototype for a class
            embedding = embedding.reshape(args.shot, args.train_way, -1).mean(dim=0)

            
            label = query_y.type(torch.LongTensor)
            distance = euclidean(model(query_x), embedding)
            prob = F.softmax(distance, dim=1)

            loss = loss_fn(prob, label)
            acc = get_accuracy(label, prob)
            print('epoch{}, {}/{}, lost={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))
            average_loss = update_avg(i + 1, average_loss, loss.item())
            average_accuracy = update_avg(i + 1, average_accuracy, acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            embedding = None
            loss = None
            distanece = None
        
        model.eval()
        average_loss_val = 0
        average_accuracy_val = 0

        # evaluate after epoch.
        with torch.no_grad():
            for i, batch in enumerate(test_loader, 1):
                num = args.shot * args.test_way
                support_x, query_x = batch[0][:num].to(device), batch[0][num:].to(device)
                support_y, query_y = batch[1][:num].to(device), batch[1][num:].to(device)
                embedding = model(support_x)
                embedding = embedding.reshape(args.shot, args.test_way, -1).mean(dim=0)

                label = query_y.type(torch.LongTensor)
                distance = euclidean(model(query_x), embedding)
                prob = F.softmax(distance, dim=1)
                
                loss = loss_fn(prob, label)
                acc = get_accuracy(label, prob)
                average_loss_val = update_avg(i + 1, average_loss_val, loss.item())
                average_accuracy_val = update_avg(i + 1, average_accuracy_val, acc)

                embedding = None
                loss = None
                distanece = None

        print("epoch {} validation: loss={:4f} acc={:4f}".format(epoch, average_loss, average_accuracy))
        if average_accuracy > training_log['max_acc']:
            training_log['max_acc'] = acc
            save_model(model, 'max-acc', args.save)

        training_log['train_loss'].append(average_loss)
        training_log['train_acc'].append(average_accuracy)
        training_log['val_loss'].append(average_loss_val)
        training_log['val_acc'].append(average_accuracy_val)

        torch.save(training_log, os.path.join(args.save, 'training_log'))
        save_model(model, 'model', args.save)

        if epoch % 1 == 0:
            save_model(model, 'model', args.save)
        
        time_b = datetime.datetime.now()
        print('ETA:{}/{}'.format((time_b - time_a).seconds, (time_b - time_a).seconds * (args.epoch - epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=200)
    parser.add_argument('-b', '--batch_size_train', type=int, default=100)
    parser.add_argument('-bt', '--batch_size_test', type=int, default=100)
    parser.add_argument('-s', '--shot', type=int, default=1)
    parser.add_argument('-q', '--query', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=30)
    parser.add_argument('--test_way', type=int, default=30)
    parser.add_argument('-sv', '--save', default='./model/proto')
    args = parser.parse_args()
    print(vars(args))
    train(args)

