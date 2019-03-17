import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import soundfile as sf

fs = 8000
y, fs = librosa.load('../features/pred/babble/0dB/DR1_MREB0_SX205.wav', sr=8000)
# y, fs = librosa.load('../features/test_wav/babble/0dB/DR1_MREB0_SX205.wav', sr=8000)
# y, fs = librosa.load('../Noise_Addition/timit_128/test/DR1_MREB0_SX205.wav', sr=8000)
print(fs)


D = librosa.core.stft(y, n_fft = 128)
D_a = np.abs(D)
max_value = np.max(D_a)
print(max_value)
D_p = np.angle(D)
D_db = librosa.core.amplitude_to_db(D_a, ref=1)
D_ia = librosa.core.db_to_amplitude(D_db, ref=1)
D_i = D_ia * (np.cos(D_p) + 1j * np.sin(D_p))
y_i = librosa.core.istft(D_i)

# D_db = (D_db - np.mean(D_db, axis=1).reshape(-1, 1)) / np.std(D_db, axis=1).reshape(-1, 1)

plt.figure()
librosa.display.specshow(D_db, y_axis='log', x_axis='time', sr=8000, cmap='jet')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

print(np.mean(D_db, axis=1))
print(np.max(D_i - D))
print(np.max(y_i - y[:len(y_i)]))

sf.write('./pred.wav', y[:], fs)

plt.show()
