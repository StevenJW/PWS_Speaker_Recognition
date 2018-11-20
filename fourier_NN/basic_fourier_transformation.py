import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import wave
import sys

audiofile = 'AudioFiles/Asteven1.wav'
spf = wave.open(audiofile,'r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
fft_out = fft(signal)


Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title(audiofile)
plt.plot(Time,np.abs(fft_out))
plt.show()
