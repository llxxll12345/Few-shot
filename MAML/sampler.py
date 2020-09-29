import torch
import numpy as np
from omniglot_d import *
from torch.utils.data import DataLoader

# N-shot task sampler
class Sampler():
    """
        Sampler to obtain k-way (k classes) n-shot (n per class) samples.
        label is the list of labels for all training samples.
        __iter__ returns indice of the sample to train on.
    """
    def __init__(self, label, n_batch, n_class, n_per_class, limit_class=False):
        self.n_batch = n_batch
        self.n_class = n_class
        self.n_per_class = n_per_class

        label = np.array(label)
        # map from a label name to a set of index
        self.index_map = []
        # Should process all the inputs here
        class_range = max(label) + 1 if not limit_class else n_class + 1
        for i in range(class_range):
            index = np.argwhere(label==i).reshape(-1)
            self.index_map.append(torch.from_numpy(index))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for b in range(self.n_batch):
            batch = []
            classes = []
            classes = torch.randperm(len(self.index_map))[:self.n_class]
            assert len(classes) == self.n_class
            for c in classes:
                ind_list = self.index_map[c]
                pos_list = torch.randperm(len(ind_list))[:self.n_per_class]
                batch.append(ind_list[pos_list])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
def test_sampler():
    dataset = Omniglot("dataset/images_background", "train", True)
    print(len(dataset.labels))
    test_sampler = Sampler(dataset.labels, 10, 30, 6, limit_class=True)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)
    test = []
    for i, batch in enumerate(test_loader, 1):
        if i == 1:
            print(len(batch))

test_sampler()