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
clean_train_folder = r'F:\timit\train'
noisy_train_folder=r'F:\ECE271B_Project\Noise_Addition\results\train'
noisy_types=[r'\babble',r'\destroyerengine',r'\factory1',r'\hfchannel']
SNRs=[r'\5db',r'\10db',r'\15db',r'\20db',r'\0db',r'\-5db']
noisy_train_folder = r'F:\ECE271B_Project\Noise_Addition\results\train'
noisy_test_folder = r'F:\ECE271B_Project\Noise_Addition\results\test'
clean_test_folder =r'F:\ECE271B_Project\Noise_Addition\results\test\babble\0db'
#noisy_test_folder =r'F:\test_results'
serialized_train_folder = r'F:\train_np_results'
serialized_test_folder = r'F:\test_np_results'
window_size = 2 ** 14  # about 1 second of samples
#sample_rate = 16000


def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


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
def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    for snr in SNRs:
        for noise in noisy_types:
            
            stride = 0.5
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
                    #clean_file = os.path.join(clean_folder, filename)
                    noisy_file = os.path.join(noisy_folder, filename)
                    # slice both clean signal and noisy signal
                    #clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
                    #noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
                    #converted_clean=saveConvert(clean_file)
                    converted_noisy=saveConvert(noisy_file)
                    # serialize - file format goes [original_file]_[slice_number].npy
                    # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
                    """
                    for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                        pair = np.array([slice_tuple[0], slice_tuple[1]])
                        np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=pair)
                    """
            
                    np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=converted_noisy)
                    #np.save(os.path.join(serialized_test_folder, '{}'.format(filename)), arr=converted_noisy)
                    max_idxs.append((filename,converted_noisy.shape[1]))
            #np.save(os.path.join(serialized_test_folder, '{}'.format('idxs')), arr=np.array(max_idxs))
            df = pd.DataFrame(max_idxs, columns=["filename","max_idx"])
            df.to_csv('list.csv', index=False)


def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break


if __name__ == '__main__':
    process_and_serialize('train')
    #data_verify('train')
    #process_and_serialize('test')
    #data_verify('test')