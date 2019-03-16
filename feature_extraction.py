#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:55:26 2019

@author: ericl
"""

import os
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import platform


if platform.system() == 'Linux':
    #These two dir are used to track the name of the files we want to convert, as only 200 test audios were selected, we track them from
    # the ones generated with matlab
    clean_train_folder = './Noise_Addition/timit_128/train'
    clean_test_folder='./Noise_Addition/timit_128/test'

    #These are the folders where we have our noisy data stored
    noisy_test_folder = './Noise_Addition/results/test'
    noisy_train_folder = './Noise_Addition/results/train'

    #output stft features in numpy form and save in below %dirs
    output_folder = './features'
    output_test_folder='./features/test'
    output_train_folder='./features/train'

    #the type of noise and SNR we want to deal with, add in dirs to ensure consistency
    noisy_types=['babble','white','alarm','destroyerengine', 'pink', 'volvo']
    test_noisy_types=['pink', 'volvo']
    SNRs=['5dB','10dB','15dB','20dB','0dB','-5dB']

else:
    #These two dir are used to track the name of the files we want to convert, as only 200 test audios were selected, we track them from
    # the ones generated with matlab
    clean_train_folder = r'.\Noise_Addition\timit_128\timit\train'
    clean_test_folder=r'.\Noise_Addition\timit_128\timit\test'

    #These are the folders where we have our noisy data stored
    noisy_test_folder = r'.\Noise_Addition\results\test'
    noisy_train_folder = r'.\Noise_Addition\results\train'

    #output stft features in numpy form and save in below dirs
    output_folder=r'.\features'
    output_test_folder=r'.\features\test'
    output_train_folder=r'.\features\train'

    #the type of noise and SNR we want to deal with, add in dirs to ensure consistency
    noisy_types=[r'\babble',r'\white',r'\factory1',r'\hfchannel']
    test_noisy_types=[r'pink', r'volvo']
    SNRs=[r'\5db',r'\10db',r'\15db',r'\20db',r'\0db',r'\-5db']
    #SNRs=[r'\5db']
    #noisy_types=[r'\babble']


window_size = 2 ** 14  # about 1 second of samples
#sample_rate = 16000


def saveConvert_info(file):
    """
    input a wav file, return np array after stft
    """
    
    y, fs = librosa.load(file, sr=8000)
    D = librosa.core.stft(y, n_fft = 128)
    #sample_rate, samples = wavfile.read(file)
    #x=scipy.signal.stft(samples,sample_rate)
    D_a = np.abs(D)
    D_db = librosa.core.amplitude_to_db(D_a, ref=np.max)
    phase=np.angle(D)
    max_value=np.max(D_a)
    return [D_db, phase, max_value]

def saveConvert_data(file):
    """
    input a wav file, return np array after stft
    """
    y, fs = librosa.load(file, sr=8000)
    D = librosa.core.stft(y, n_fft = 128)
    D_a = np.abs(D)

    D_db = librosa.core.amplitude_to_db(D_a, ref=np.max)
    return D_db
    

def normalize(data):
    """
    normalize data by each row
    
    intype: np array (n_fft // 2 + 1) * n
    rtype: np array (n_fft // 2 + 1) * n
    
    """
    #this function should not be utilized until we get the mean and std of our data
    return (data-np.mean(data,axis=1).reshape(-1, 1)) / np.std(data,axis=1).reshape(-1, 1)

test_dict={}
def processData(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    count=0

    #Generate features for clean data
    max_indices = []
    if data_type == 'train':
        output_clean_folder = os.path.join(output_train_folder, 'clean')
        if not os.path.exists(output_clean_folder):
            os.makedirs(output_clean_folder)
        for root, dirs, files in os.walk(clean_train_folder):
            for filename in tqdm(files, desc='Converting {} audios'.format(data_type)):
                if '.wav' in filename:
                    clean_file = os.path.join(clean_train_folder, filename)
                    data = saveConvert_data(clean_file)
                    np.save(os.path.join(output_clean_folder, '{}'.format(filename)), data)
                    max_indices.append((filename, data.shape[1]))
        df = pd.DataFrame(max_indices, columns=["filename","max_idx"])
        df.to_csv(os.path.join(output_folder, 'list.csv'), index=False)
    
    num_features, _ = data.shape
    mean = np.zeros([num_features, 1])
    
    for snr in SNRs:
        for noise in noisy_types:
            
            if data_type == 'train':
                clean_folder = clean_train_folder
                noisy_folder = os.path.join(noisy_train_folder, noise, snr)
                serialized_folder = os.path.join(output_train_folder, noise, snr)
                
                if noise in test_noisy_types:
                    continue
            else:
                clean_folder = clean_test_folder
                noisy_folder = os.path.join(noisy_test_folder, noise, snr)
                serialized_folder = os.path.join(output_test_folder, noise, snr)
            if not os.path.exists(serialized_folder):
                os.makedirs(serialized_folder)
            
            for root, dirs, files in os.walk(clean_folder):
                for filename in tqdm(files, desc='Converting {} audios'.format(data_type)):
                    if '.wav' in filename:
                        noisy_file = os.path.join(noisy_folder, filename)
                        
                        if not os.path.isfile(noisy_file):
                            continue
                        
                        #get the mean
                        if data_type == 'train':
                            converted_noisy=saveConvert_data(noisy_file)
                            mean += np.sum(converted_noisy,axis=1).reshape(-1, 1)

                            count += len(converted_noisy[0])

                            np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=converted_noisy)
                        else:
                            data, phase, max_value = saveConvert_info(noisy_file)
                            data_info = {}
                            data_info['data'] = data
                            data_info['phase'] = phase
                            data_info['max_value'] = max_value
                            np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=data_info)
                            
    mean = mean / count
    np.save('mean.npy',mean)
            


#get the mean and std for each feature, and then feed in normalized ones only in the traininig process, done by pytorch
def get_std(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    mean=np.load('mean.npy').reshape(-1,1)
    std=np.zeros(mean.shape)
    count=0
    for snr in SNRs:
        for noise in noisy_types:
            
            if data_type == 'train':
                clean_folder = clean_train_folder
                noisy_folder = os.path.join(noisy_train_folder, noise, snr)
                serialized_folder = os.path.join(output_train_folder, noise, snr)
                
                if noise in test_noisy_types:
                    continue
            else:
                clean_folder = clean_test_folder
                noisy_folder = os.path.join(noisy_test_folder, noise, snr)
                serialized_folder = os.path.join(output_test_folder, noise, snr)
            if not os.path.exists(serialized_folder):
                os.makedirs(serialized_folder)
            
            #clean_folder = clean_test_folder
            #noisy_folder = noisy_test_folder+noise+snr
            
            """
            #this piece of code is used to generate converted data along with their phases,angle,etc.
            for root, dirs, files in os.walk(clean_folder):
                if len(files) == 0:
                    continue
                #print('current folder',dirs)
                for filename in tqdm(files, desc='Converting {} audios'.format(data_type)):
                    noisy_file = os.path.join(noisy_folder, filename)
                    if '.wav' in filename:
                    #[phase,mean,std,max_value]


                        data=saveConvert_data(noisy_file)
                        data=normalize(data)
                        (a,b,c,d)=saveConvert_info(noisy_file)
                        test_dict['phase']=a
                        test_dict['mean']=b
                        test_dict['std']=c
                        test_dict['max_value']=d
                        test_dict['data']=data
                        np.save(os.path.join(serialized_folder, '{}'.format(filename)),test_dict)
                        #np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=test_dict)
                        #print(noisy_file)

                        
            """
            
            for root, dirs, files in os.walk(clean_folder):
                for filename in tqdm(files, desc='Converting {} audios'.format(data_type)):
                    if '.wav' in filename:
                        noisy_file = os.path.join(noisy_folder, filename)
                        converted_noisy=saveConvert_data(noisy_file)
                        
                        #get the mean
                        std += np.sum((converted_noisy-mean)**2,axis=1).reshape(-1, 1)
                        
                        count+=len(converted_noisy[0])
                            
                        #normalization of data will be performed at the training stage
                        #test=normalize(converted_noisy)
                        #np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=converted_noisy)
                        #print('saving dir',serialized_folder)
            

                    #print(np.sum(np.isnan(converted_noisy)))
                    #max_idxs.append((filename,converted_noisy.shape[1]))
    std = std / count
    np.save('std.npy',std ** 0.5)


if __name__ == '__main__':
    processData('train')
    get_std('train')

