import os
import torch
import numpy as np
import shutil

def cos_sim(a, b):
    """
        cosine similarity
    """
    return np.dot(a, b)/(np.norm(a) * np.norm(b))

def l2_loss(y, pred):
    return ((pred-y)**2).sum() / len(pred * 2) 

def update_avg(n, avg, x):
    return (avg * n + x) / (n + 1)

def get_accuracy(y, pred):
    return (torch.argmax(pred, dim=1) == y).sum().item()/len(pred)

def euclidean(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return -((a - b)**2).sum(dim=2)

def renew_path(path):
    if os.path.exists(path):
        rm = input('{} exists, remove? y/n'.format(path))
        if rm != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)



