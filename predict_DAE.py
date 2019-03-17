import torch
import numpy as np
import os
import librosa
import scipy
import soundfile as sf
from tqdm import tqdm

num_frames = 11
window_size = num_frames // 2
fs = 8000
n_fft = 128

root_dir = './features'
model_fn = './saved_model/DAE.pt'

noise_types = ['babble', 'destroyerengine',
               'alarm', 'volvo', 'white', 'pink']
SNRs = ['-5dB', '0dB', '5dB',
        '10dB', '15dB', '20dB']

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

mean = np.load('./mean.npy')
std = np.load('./std.npy')

with torch.no_grad():
    for snr in SNRs:
        for noise_type in noise_types:
            noisy_dir = os.path.join(root_dir, 'test', noise_type, snr)
            print(noisy_dir)
            for root, dirs, files in os.walk(noisy_dir):
                for fn in tqdm(files):
                    fn_base = fn.split('.')[0]
                    audioinfo = np.load(os.path.join(root, fn))
                    data = audioinfo.item()['data']
                    D_p = audioinfo.item()['phase']
                    max_value = audioinfo.item()['max_value']
                    # print(data.shape)
                    # print(mean.shape)
                    # print(std.shape)
                    # print(phase.shape)
                    # print(max_value)

                    _, length = data.shape
                    data = (data - mean) / std
                    data = torch.tensor(data).float()
                    data_pred = torch.zeros(data.shape)
                    for i in range(window_size, length - window_size, num_frames):
                        data_in = data[:, i - window_size: i + window_size + 1].contiguous().view(1, -1)
                        data_pred[:, i - window_size: i + window_size + 1] = model(data_in).view(-1, num_frames)

                    data_pred = data_pred.numpy()
                    data_pred = data_pred[:, : length - window_size]
                    D_p = D_p[:, : length - window_size]
                    data_pred = data_pred * std + mean
                    D_a = librosa.core.db_to_amplitude(data_pred, ref=1)
                    D = D_a * (np.cos(D_p) + 1j * np.sin(D_p))

                    y_pred = librosa.istft(D)
                    out_dir = os.path.join(root_dir, 'pred', noise_type, snr)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_fn = os.path.join(out_dir, fn_base + '.wav')
                    sf.write(out_fn, y_pred, fs)
