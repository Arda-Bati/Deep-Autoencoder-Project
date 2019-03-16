import torch
import numpy as np
import os
import librosa
import scipy
from tqdm import tqdm

num_frames = 11
window_size = num_frames // 2
fs = 8000
n_fft = 256

root_dir = './dataset/test'
model_fn = './Models/network.pt'

noise_types = ['babble', 'destroyerengine',
               'factory1', 'hfchannel', 'white', 'pink']
SNRs = ['-5db', '0db', '5db',
        '10db', '15db', '20db']

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")


model = torch.load(model_fn)
model = model.cpu()
# model = model.to(computing_device)

with torch.no_grad():
    for snr in SNRs:
        for noise_type in noise_types:
            noisy_dir = os.path.join(root_dir, 'noisy_normalized', noise_type, snr)
            print(noisy_dir)
            for root, dirs, files in os.walk(noisy_dir):
                for fn in tqdm(files):
                    fn_base = fn.split('.')[0]
                    audioinfo = np.load(os.path.join(root, fn))
                    data = audioinfo[()]['data']
                    D_p = audioinfo[()]['phase']
                    mean = audioinfo[()]['mean']
                    std = audioinfo[()]['std']
                    max_value = audioinfo[()]['max_value']
                    # print(data.shape)
                    # print(mean.shape)
                    # print(std.shape)
                    # print(phase.shape)
                    # print(max_value)

                    _, length = data.shape
                    data = torch.tensor(data)
                    data_pred = torch.zeros(data.shape)
                    for i in range(window_size, length - window_size):
                        data_in = data[:, i - window_size: i + window_size + 1].contiguous().view(1419)
                        data_pred[:, i] = model(data_in)

                    data_pred = data_pred.numpy()
                    data_pred = data_pred[:, window_size: length - window_size]
                    D_p = D_p[:, window_size: length - window_size]
                    data_pred = data_pred * std.reshape(-1, 1) + mean.reshape(-1, 1)
                    D_a = librosa.core.db_to_amplitude(data_pred, ref=max_value)
                    D = D_a * (np.cos(D_p) + 1j * np.sin(D_p))

                    _, y_pred = scipy.signal.istft(D, fs=fs)
                    y_pred = y_pred / np.max(np.abs(y_pred))
                    out_fn = os.path.join(root_dir, 'noisy_pred', noise_type, snr, fn_base + '.wav')
                    scipy.io.wavfile.write(out_fn, rate=fs, data=y_pred)
