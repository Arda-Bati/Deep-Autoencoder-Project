# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:24:29 2019

@author: ericl
"""

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read(r'F:\timit\train\DR1_FCJF0_SA2.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()