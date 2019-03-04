# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:55:26 2019

@author: ericl
"""

import os

import librosa
import numpy as np
from tqdm import tqdm

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd

# fit this to your own dir
clean_train_folder = r'F:\timit\train'
clean_test_folder =r'F:\timit\test'

#This script is for the preparation of the npy files of the audios, which after STFT and normalization

#if generating noisy files
#noisy_types=[r'\babble',r'\destroyerengine',r'\factory1',r'\hfchannel']
#SNRs=[r'\5db',r'\10db',r'\15db',r'\20db',r'\0db',r'\-5db']
#noisy_train_folder = r'F:\ECE271B_Project\Noise_Addition\results\train'
#noisy_test_folder = r'F:\ECE271B_Project\Noise_Addition\results\test'


#if generating clean files
noisy_types=[r""]
SNRs=[r""]
noisy_train_folder = r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\train'
noisy_test_folder = r'F:\ECE271B_Project\Noise_Addition\timit_128\timit\test'

clean_test_folder =r'F:\ECE271B_Project\Noise_Addition\results\test\babble\0db'


#this is where to put the generated files
serialized_train_folder = r'F:\train_np_results_clean\clean'
serialized_test_folder = r'F:\test_np_results'




def saveConvert(file):
    """
    input a wav file, return np array after stft
    """
    sample_rate, samples = wavfile.read(file)
    x=scipy.signal.stft(samples,sample_rate)
    D_a = np.abs(x[2])

    D_db = librosa.core.amplitude_to_db(D_a, ref=np.max)
    return D_db


max_idxs=[]


def normalize(data):
    return (data-np.mean(data,axis=1).reshape(129,1))/np.std(data,axis=1).reshape(129,1)
def process_and_serialize(data_type):
    """
    Convert wav files into npy array after stft convertion
    """
    for snr in SNRs:
        for noise in noisy_types:
            max_idxs=[]
            if data_type == 'train':
                clean_folder = clean_train_folder
                noisy_folder = noisy_train_folder+noise+snr
                serialized_folder = serialized_train_folder+noise+snr
            else:
                clean_folder = clean_test_folder
                noisy_folder = noisy_test_folder+noise+snr
                serialized_folder = serialized_test_folder+noise+snr
            if not os.path.exists(serialized_folder):
                os.makedirs(serialized_folder)
        
            # walk through the path, slice the audio file, and save the serialized result
            for root, dirs, files in os.walk(clean_folder):
                if len(files) == 0:
                    continue
                for filename in tqdm(files, desc='Converting {} audios'.format(data_type)):
                    noisy_file = os.path.join(noisy_folder, filename)
                    converted_noisy=saveConvert(noisy_file)
                    test=normalize(converted_noisy)
                    np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=test)

                    #print(np.sum(np.isnan(converted_noisy)))
                    max_idxs.append((filename,converted_noisy.shape[1]))
            #np.save(os.path.join(serialized_test_folder, '{}'.format('idxs')), arr=np.array(max_idxs))
            df = pd.DataFrame(max_idxs, columns=["filename","max_idx"])
            df.to_csv('list.csv', index=False)



if __name__ == '__main__':
    process_and_serialize('train')
    #data_verify('train')
    #process_and_serialize('test')
    #data_verify('test')