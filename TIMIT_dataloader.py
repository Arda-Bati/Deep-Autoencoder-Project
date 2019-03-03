# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader


from torchvision import transforms, utils

from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


class TIMITDataset(Dataset):

    def __init__ (self, window_size):

        self.noisy_data_dir = r'F:\train_np_results'
        self.clean_data_dir = r'F:\train_np_results'
        self.data_info = pd.read_csv(r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\list.csv')
        self.data_filenames = self.data_info['filename']
<<<<<<< HEAD
        self.noisetypes = {0: 'babble', 1: 'destroyerengine',
                           2: 'factory1', 3: 'hfchannel'}
        self.SNR = {0: '-5db', 1: '0db', 2: '5db',
                    3: '10db', 4: '15db', 5: '20db'}
=======

        self.noisetypes = {0: r'babble', 1: r'destroyerengine',
                           2: r'factory1', 3: r'hfchannel'}
        self.SNR = {0: r'-5db', 1: r'0db', 2: r'5db',
                    3: r'10db', 4: r'15db', 5: r'20db'}
>>>>>>> 0ce316703ed2a1a6bcd001a6ba7b6500e02d5eac
        self.window_size = window_size

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, ind):
<<<<<<< HEAD

        spec_clean = np.load(os.path.join(self.data_dir,
                                             'clean',
                                             self.data_filenames.ix[ind[0]] + '.npy'))

        spec_noisy = np.load(os.path.join(self.data_dir,
                                             'noisy',
=======
        """
   
        spec_clean = np.load(os.path.join(self.clean_data_dir,
                                             'clean',
                                             self.data_filenames.ix[ind[0]]+r'.npy'))
        """

        spec_noisy = np.load(os.path.join(self.noisy_data_dir,
>>>>>>> 0ce316703ed2a1a6bcd001a6ba7b6500e02d5eac
                                             self.noisetypes[ind[1]],
                                             self.SNR[ind[2]],
                                             self.data_filenames.ix[ind[0]] +r'.npy'))

<<<<<<< HEAD
        input = spec_noisy[:, ind[3] - self.window_size: ind[3] + self.window_size + 1]
        target = spec_clean[:, ind[3]]

        input = torch.from_numpy(input)
        input = input.contiguous().view(129 * (2 * self.window_size + 1))
        target = torch.from_numpy(target)
        target = target.contiguous().view(129)

        return (input, target)
=======
        #print(spec_noisy.shape)
        input = spec_noisy[:, ind[3] - self.window_size: ind[3] + self.window_size + 1]
        #target = spec_clean[:, ind[3]]
        input=np.reshape(input,(input.size))
        #target=np.reshape(target,(target.size))
        

        return input,input
>>>>>>> 0ce316703ed2a1a6bcd001a6ba7b6500e02d5eac


def prepareTIMIT_train(batch_size = 1, num_frame = 11, extras={}):

    window_size = num_frame // 2
    dataset = TIMITDataset(window_size)

    dataset_size = len(dataset)
    max_indices = dataset.data_info['max_idx']
    indices = []
    for i in range(100):
        for j in range(len(dataset.noisetypes)):
            for k in range(len(dataset.SNR)):
                for l in range(window_size, max_indices.ix[i] - window_size):
                    indices.append((i, j, k, l))

    sample = SubsetRandomSampler(indices)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample, num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader

if __name__ == '__main__':
    train_loader = prepareTIMIT_train(batch_size=20,
                                      num_frame=11)
<<<<<<< HEAD

    for minibatch_count, (inputs, targets) in enumerate(tqdm(train_loader), 0):
        # print(inputs.shape)
        pass

=======
    """
    for minibatch_count, (inputs,target) in enumerate(tqdm(train_loader), 0):
        #print(inputs.shape)
        print(target.shape)
        
    
>>>>>>> 0ce316703ed2a1a6bcd001a6ba7b6500e02d5eac
    print(inputs.shape)
    """
# print(targets.shape)