import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pathlib import Path
import numpy as np
import wave
import sys


x = 1
x_train = np.empty(shape=[0, 800])
y_train = np.empty(shape=[0, 2])
while True:
	audiofile = 'AudioFiles/Ematthijs'+ str(x) + '.wav'
	x += 1
	if Path(audiofile).is_file():
		spf = wave.open(audiofile,'r')
	else: 
		break

	#Extract Raw Audio from Wav File
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	fs = spf.getframerate()
	# Time and fft
	fft_out = fft(signal)
	Time=np.linspace(0, len(signal)/fs, num=len(signal))

	# Get corresponding y values
	#line2d = plt.plot(Time,np.abs(fft_out))
	xvalues = Time#line2d[0].get_xdata()
	yvalues = np.abs(fft_out)#line2d[0].get_ydata()

	goodvalues = int(181855 / 4)

	# Create new array for new graph of values
	yArrayValues = []
	for i in range(0, int(goodvalues*0.14), 4):
		idx = np.where(xvalues==xvalues[i])
		yArrayValues.append(yvalues[idx]/(1* (10**7)))
		if i == 3200:
			break


	xarray = range(len(yArrayValues))
	plt.plot(xarray, yArrayValues)
	print(yArrayValues)
	#x_train = np.append(x_train, [yArrayValues], axis=0)
	#y_train = np.append(y_train, [[1, 0]], axis=0)

plt.show()
print(y_train)
