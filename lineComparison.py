import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import wave
import sys

# Needed so the original time plots don't show up in the actual graph
lineCollection = []
for x in range(1,3):
	audiofile = 'AudioFiles/Esteven'+ str(x) + '.wav'
	spf = wave.open(audiofile,'r')

	#Extract Raw Audio from Wav File
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	fs = spf.getframerate()
	# Time and fft
	fft_out = fft(signal)
	Time=np.linspace(0, len(signal)/fs, num=len(signal))

	# Get corresponding y values
	line2d = plt.plot(Time,np.abs(fft_out))
	xvalues = line2d[0].get_xdata()
	yvalues = line2d[0].get_ydata()

	goodvalues = int(181855 / 4)
	#idx = np.where(xvalues==xvalues[goodvalues])
	#print(yvalues[idx])

	plswork = []
	#int(goodvalues*0.015)
	for i in range(0, int(goodvalues*0.14), 4):
		idx = np.where(xvalues==xvalues[i])
		plswork.append(yvalues[idx]/(1* (10**7)))

	#print(len(plswork))

	xarray = range(len(plswork))
	lineCollection.append(list(xarray))
	lineCollection.append(list(plswork))
	print(x)
	#plt.plot(xarray, plswork)

plt.gcf().clear()

plt.plot(lineCollection[0],lineCollection[1])
plt.plot(lineCollection[2],lineCollection[3])

plt.figure(1)
plt.title("Line comparison")
plt.show()
