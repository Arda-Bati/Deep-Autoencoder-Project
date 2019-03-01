# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Other libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TIMITDataset(Dataset):

    def __init__ (self,):

        self.data_dir = './dataset/train'
        self.data_info = pd.read_csv('./dataset/info.csv')
        self.data_filenames = self.data_info['filename']
        self.noisetypes = {0: 'babble', 1: 'destroyerops',
                           2: 'factory1', 3: 'hfchannel'}
        self.SNR = {0: '-5db', 1: '0db', 2: '5db',
                    3: '10db', 4: '15db', 5: '20db'}
        self.window_size = 5

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, ind):

        spec_clean = torch.load(self.data_dir,
                                'clean',
                                self.data_filenames.ix[ind[0]])

        spec_noisy = torch.load(self.data_dir,
                                'noisy',
                                self.noisetypes[ind[1]],
                                self.SNR[ind[2]]
                                self.data_filenames.ix[ind[0]])

        input = spec_noisy[:, ind[0] - window_size: ind[0] + window_size + 1]
        target = spec_clean[:, ind[0]]

        return (input, target)


def prepareTIMIT_train(batch_size = 1, num_frame = 11, extras={}):

    window_size = num_frame // 2
    dataset = TIMITDataset(window_size)

    dataset_size = len(dataset)
    indices = []
    for i in range(dataset_size):
        for j in range(len(dataset.noisetypes)):
            for k in range(len(dataset.SNR)):
                for l in range(window_size, max_indices[i] - window_size):
                    indices.append((i, j, k, l))

    sample = SubsetRandomSampler(indices)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader
