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
import platform


class TIMITDataset(Dataset):

    def __init__ (self, window_size):

        if platform.system() == 'Linux':
            self.noisy_data_dir = './features/train'
            self.clean_data_dir = './features/train/clean'
            self.data_info = pd.read_csv('./features/list.csv')
            self.data_filenames = self.data_info['filename']


            self.noisetypes = {0: 'babble', 1: 'white',
                               2: 'destroyerengine', 3: 'alarm'}
            self.SNR = {0: '-5dB', 1: '0dB', 2: '5dB',
                        3: '10dB', 4: '15dB', 5: '20dB'}
            self.mean = np.load('./mean.npy')
            self.std = np.load('./std.npy')
        
        else:
            self.noisy_data_dir = r'F:\train_np_results_clean'
            self.clean_data_dir = r'F:\train_np_results_clean\clean'
            self.data_info = pd.read_csv(r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\list.csv')
            self.data_filenames = self.data_info['filename']


            self.noisetypes = {0: r'babble', 1: r'destroyerengine',
                               2: r'factory1', 3: r'hfchannel'}
            self.SNR = {0: r'-5db', 1: r'0db', 2: r'5db',
                        3: r'10db', 4: r'15db', 5: r'20db'}
        self.window_size = window_size

    def __len__(self):
        return len(self.data_filenames)
    
    def normalize(self,data):
        return (data-np.mean(data))/np.std(data)

    def __getitem__(self, ind):

   
        spec_clean = np.load(os.path.join(self.clean_data_dir,
                                             self.data_filenames.ix[ind[0]]+r'.npy'))


        spec_noisy = np.load(os.path.join(self.noisy_data_dir,
                                             self.noisetypes[ind[1]],
                                             self.SNR[ind[2]],
                                             self.data_filenames.ix[ind[0]] +r'.npy'))
        
        
        # where_are_NaNs = np.isnan(spec_clean)
        # spec_clean[where_are_NaNs] = 0

        input = spec_noisy[:, ind[3] - self.window_size: ind[3] + self.window_size + 1]
        target = spec_clean[:, ind[3] - self.window_size: ind[3] + self.window_size + 1]
        
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        input=np.reshape(input,(input.size))
        target=np.reshape(target,(target.size))
        
        return input,target

def prepareTIMIT_train(batch_size = 1, 
                       num_frames = 11,
                       shuffle = True,
                       seed = 1,
                       extras={}):

    window_size = num_frames // 2
    dataset = TIMITDataset(window_size)

    dataset_size = len(dataset)
    max_indices = dataset.data_info['max_idx']
    
    all_indices = list(range(dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    train_split = 800
    val_split = 50
    train_ind, val_ind = all_indices[val_split : val_split + train_split], all_indices[: val_split]
    
    train_indices = []
    for i in train_ind:
        for j in range(len(dataset.noisetypes)):
            for k in range(len(dataset.SNR)):
                for l in range(window_size, max_indices.ix[i] - window_size, window_size):
                    train_indices.append((i, j, k, l))

    val_indices = []
    for i in val_ind:
        for j in range(len(dataset.noisetypes)):
            for k in range(len(dataset.SNR)):
                for l in range(window_size, max_indices.ix[i] - window_size, window_size):
                    val_indices.append((i, j, k, l))

    sample_train = SubsetRandomSampler(train_indices)
    sample_val = SubsetRandomSampler(val_indices)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)
    
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = prepareTIMIT_train(batch_size=200,
                                                  num_frames=11,
                                                  shuffle=True,
                                                  seed=1,
                                                  extras={"num_workers": 0, "pin_memory": True})

    for minibatch_count, (inputs, targets) in enumerate(tqdm(val_loader), 0):
        # print(inputs.shape)
        # print(np.sum(np.isnan(targets.numpy())))
        pass
