import torch
import numpy as np
from omniglot import *

class Sampler():
    """
        Sampler to obtain k-way (k classes) n-shot (n per class) samples.
        label is the list of labels for all training samples.
        __iter__ returns indice of the sample to train on.
    """
    def __init__(self, label, n_batch, n_class, n_per_class, shuffle_class=True):
        self.n_batch = n_batch
        self.n_class = n_class
        self.n_per_class = n_per_class
        self.shuffle_class = shuffle_class

        label = np.array(label)
        # map from a label name to a set of index
        self.index_map = []
        # Should process all the inputs here
        for i in range(max(label) + 1):
            index = np.argwhere(label==i).reshape(-1)
            self.index_map.append(torch.from_numpy(index))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for b in range(self.n_batch):
            batch = []
            classes = []
            # shuffle_class the class list, and pick the first n_class classes
            if self.shuffle_class:
                classes = torch.randperm(len(self.index_map))[:self.n_class]
            else:
                classes = [i for i in range(len(self.index_map))][:self.n_class]
            assert len(classes) == self.n_class
            for c in classes:
                # pick the data indice of that class
                ind_list = self.index_map[c]
                # pick n samples (shot + query) from each class
                pos_list = torch.randperm(len(ind_list))[:self.n_per_class]
                # append the indice of the samples
                batch.append(ind_list[pos_list])
            # stack(): stack all the tensors in the array to a single tensor
            # t(): transpose of the tensor
            # output array class type: [class1, class2, class3, class1, class2 ...]
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
def test_sampler():
    dataset = OmiglotSet('train')
    print(len(dataset.labelSet))
    test_sampler = Sampler(dataset.label, 100, 30, 1)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)
    for i, batch in enumerate(test_loader, 1):
        print(np.array(batch[1]).shape)
        #print(i, batch)

#test_sampler()