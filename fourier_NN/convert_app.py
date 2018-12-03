import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pathlib import Path
import numpy as np
import wave
import sys
import pickle



audiofile = '../AudioFiles/Esteven1.wav'
spf = wave.open(audiofile,'r')
print(spf.__dict__)

#audiofile = '../AudioFiles/'+ str(vowel) + str(audioFileName) + str(x) + '.wav'
#spf = wave.open(content,'r')
"""
#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
# Time and fft
fft_out = fft(signal)
Time=np.linspace(0, len(signal)/fs, num=len(signal))
# Get corresponding y values
xvalues = Time
yvalues = np.abs(fft_out)
goodvalues = int(181855 / 4)
# Create new array for new graph of values
yArrayValues = []
for i in range(0, int(goodvalues*0.14), 8):
	idx = np.where(xvalues==xvalues[i])
	yArrayValues.extend(yvalues[idx]/(1* (10**7)))
	if i >= 3996:
		break
print(yArrayValues)
"""