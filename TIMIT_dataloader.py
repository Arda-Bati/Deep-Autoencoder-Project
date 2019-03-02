# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TIMITDataset(Dataset):

    def __init__ (self,):

        self.noisy_data_dir = r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\train'
        self.clean_data_dir = r'F:\train_np_results'
        self.data_info = pd.read_csv(r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\list.csv')
        self.data_filenames = self.data_info['filename']
        self.noisetypes = {0: r'babble', 1: r'destroyerengine',
                           2: r'factory1', 3: r'hfchannel'}
        self.SNR = {0: r'-5db', 1: r'0db', 2: r'5db',
                    3: r'10db', 4: r'15db', 5: r'20db'}
        self.window_size = 5

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, ind):

        spec_clean = torch.load(self.clean_data_dir,
                                self.data_filenames.ix[ind[0]])

        spec_noisy = torch.load(self.noisy_data_dir,
                                self.noisetypes[ind[1]],
                                self.SNR[ind[2]],
                                self.data_filenames.ix[ind[0]])

        input = spec_noisy[:, ind[3] - self.window_size: ind[3] + self.window_size + 1]
        target = spec_clean[:, ind[3]]

        return (input, target)


def prepareTIMIT_train(batch_size = 1, num_frame = 11, extras={}):

    window_size = num_frame // 2
    dataset = TIMITDataset()

    dataset_size = len(dataset)
    max_indices = dataset.data_info['max_idx']
    indices = []
    #"""
    #f=open('test.txt','w')
    for i in range(dataset_size):
        #print(i,dataset_size)
        for j in range(len(dataset.noisetypes)):
            for k in range(len(dataset.SNR)):
                for l in range(window_size, max_indices.ix[i] - window_size):
                    indices.append((i, j, k, l))
                    #f.write(str((i,j,k,l)))
                    #f.writelines(str((i,j,k,l)))
    #f.close()
    #"""
    sampler = SubsetRandomSampler(indices)
    print(sampler)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader
