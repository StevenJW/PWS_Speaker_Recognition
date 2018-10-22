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




'''import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import wave
import sys

spf = wave.open('AudioFiles/Asteven1.wav','r')

#Extract Raw Audio from Wav File
signalv = spf.readframes(-1)
signalv = np.fromstring(signalv, 'Int16')
fs = spf.getframerate()

fft_out = signal.stft(signalv)

fs = 3
N = 3
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
Time = np.arange(N) / float(fs)
#Time=np.linspace(0, len(signalv)/fs, num=len(signalv))


plt.figure(1)
plt.title('Signal Wave...')
plt.plot(Time,np.abs(fft_out))
plt.show()'''
