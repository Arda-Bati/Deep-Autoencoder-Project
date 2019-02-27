import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, fs = librosa.load('./timit/train/DR1_FCJF0_SA1.wav', sr=8000)
print(y.shape)

D = librosa.core.stft(y, n_fft = 2048)
D_a = np.abs(D)
D_p = np.angle(D)
D_db = librosa.core.amplitude_to_db(D_a)
D_power = librosa.core.db_to_power(D_db)
D_idb = librosa.core.power_to_db(D_power)
D_ia = librosa.core.db_to_amplitude(D_idb)
D_i = D_ia * (np.cos(D_p) + 1j * np.sin(D_p))

plt.figure()
librosa.display.specshow(D_db, y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.figure()
librosa.display.specshow(D_power, x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

print(np.max(D_i - D))


D_t = torch.from_numpy(D_power)
print(D_t.size())
torch.save(D_t, './DR1_FCJF0_SA1.pt')
plt.show()
