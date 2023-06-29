import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import pandas as pd
import numpy as np
import pickle, tqdm, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def __train_test_split(dataset, dev_size, test_size, seed, stratify, **kwargs):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    dev_split = dev_size+test_size
    test_split = test_size/dev_split

    train_indices, dev_indices = train_test_split(indices, test_size=dev_split, random_state=seed, stratify=dataset.labels if stratify else None, **kwargs)
    dev_indices, test_indices = train_test_split(dev_indices, test_size=test_split, random_state=seed, stratify=dataset.labels[dev_indices] if stratify else None, **kwargs)

    return train_indices, dev_indices, test_indices


def torch_train_test_split(dataset, batch_train, batch_dev, batch_test, dev_size=.2, test_size=.1, seed=None, stratify=True, **kwargs):
    # Creating data indices for training and validation splits:
    train_indices, dev_indices, test_indices = __train_test_split(dataset, dev_size, test_size, seed, stratify=stratify)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler, **kwargs)
    dev_loader   = DataLoader(dataset, batch_size=batch_dev, sampler=dev_sampler, **kwargs)
    test_loader  = DataLoader(dataset, batch_size=batch_test, sampler=test_sampler, **kwargs)

    return train_loader, dev_loader, test_loader


def timeseries_train_test_split(series, dev_size=.2, test_size=.1, seed=100, stratify=None, **kwargs):
    # Split Trajectories
    train_indices, dev_indices, test_indices = __train_test_split(series, dev_size=dev_size, test_size=test_size,
                                                                  seed=seed, stratify=stratify, **kwargs)
    # Return Trajectories' Datasets
    return train_indices, dev_indices, test_indices


class VRFDataset(Dataset):
    def __init__(self, data, scaler=None, max_dt=1920, dtype=np.float32):
        self.samples = data['samples'].values
        self.labels = data['labels'].values.ravel()
        self.lengths = [len(l) for l in self.samples]

        self.max_dt = max_dt
        self.dtype = dtype
        
        if scaler is None:
            self.scaler = StandardScaler().fit(np.concatenate(self.samples))
        else:
            self.scaler = scaler

    def pad_collate(self, batch):
        '''
        xx: Samples (Delta Trajectory), yy: Labels (Next Delta), ll: Lengths
        '''
        (xx, yy, ll) = zip(*batch)
        
        # Right Zero Padding with Zeroes (for delta trajectory)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, torch.stack(yy), torch.tensor(ll)
    
    def __getitem__(self, item):
        return torch.tensor(self.scaler.transform(self.samples[item]).astype(self.dtype)),\
               torch.tensor(self.labels[item].reshape(1, -1).astype(self.dtype)),\
               torch.tensor(self.lengths[item])

    def __len__(self):
        return len(self.labels)
