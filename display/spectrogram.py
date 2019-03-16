import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy

fs = 8000
# y, fs = librosa.load('./timit/test/DR1_FAKS0_SI2203.wav', sr=8000)
# y, fs = librosa.load('./dataset/test/noisy/white/-5db/DR1_FAKS0_SI2203.wav', sr=8000)
# y, fs = librosa.load('./dataset/test/noisy_pred/white/-5db/DR1_FAKS0_SI2203.wav', sr=8000)
y, fs = librosa.load('../evaluator/test.wav', sr=8000)
# fs, y = scipy.io.wavfile.read('./timit/test/DR1_FAKS0_SA1.wav')
# y, fs = librosa.load('./results/train/babble/0db/DR1_FCJF0_SA1.wav', sr=8000)
print(fs)

# _, _, D = scipy.signal.stft(y, fs=fs)
# _, y_i = scipy.signal.istft(D, fs=fs)
# y_i = y_i / np.max(np.abs(y_i))
# scipy.io.wavfile.write('./test.wav', rate = fs, data = np.round(y_i).astype('int'))

# y_i = librosa.core.istft(D)
# librosa.output.write_wav('./test.wav', y=y_i, sr=fs)

D = librosa.core.stft(y, n_fft = 256)
D_a = np.abs(D)
max_value = np.max(D_a)
print(max_value)
D_p = np.angle(D)
D_db = librosa.core.amplitude_to_db(D_a, ref = np.max)
# D_power = librosa.core.db_to_power(D_db)
# D_idb = librosa.core.power_to_db(D_power)
D_ia = librosa.core.db_to_amplitude(D_db, ref = max_value)
D_i = D_ia * (np.cos(D_p) + 1j * np.sin(D_p))
y_i = librosa.core.istft(D_i)

# D_db = (D_db - np.mean(D_db, axis=1).reshape(-1, 1)) / np.std(D_db, axis=1).reshape(-1, 1)

plt.figure()
librosa.display.specshow(D_db, y_axis='log', x_axis='time', sr=8000, cmap='jet')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

# plt.figure()
# librosa.display.specshow(D_power, x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()

print(np.max(D_i - D))
print(np.max(y_i - y[:len(y_i)]))

plt.show()
