import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pathlib import Path
import numpy as np
import wave
import sys
import pickle

fileName1 = 'steven'
fileName2 = 'matthijs'
z = 50
type-multi = True
testFile = False

if type-multi == True:
	if testFile == True:
		storeFile = 'store_1_multi-test'
	else:
		storeFile = 'store_1_multi'
else:
	if testFile == True:
		storeFile = 'store_1_sigmoid-test'
	else:
		storeFile = 'store_1_sigmoid'


def convert_train_data(audioVowel):
	global fileName1
	global fileName2
	temp_train = []
	list_train = []
	if type-multi == True:
		list_train, y_data = convert_train_data_from(fileName1, audioVowel, list_train)
		temp_train.extend(y_data)

	list_train, y_data = convert_train_data_from(fileName2, audioVowel, list_train)
	temp_train.extend(y_data)
	return np.array(list_train), np.array(temp_train)
	
def convert_train_data_from(audioFileName, vowel, data_list):
	global fileName2
	global z
	x = 1
	#temp2_train = np.empty(shape=1)
	temp2_train = []
	while True:
		print(x)
		'''if x == 4 and audioFileName == fileName2:
			x += 1
		if x == 13 and audioFileName == fileName2:
			x += 2'''
		audiofile = 'AudioFiles/'+ str(vowel) + str(audioFileName) + str(x) + '.wav'
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
		xvalues = Time
		yvalues = np.abs(fft_out)

		goodvalues = int(181855 / 4)

		# Create new array for new graph of values
		yArrayValues = []
		for i in range(0, int(goodvalues*0.14), 4):
			idx = np.where(xvalues==xvalues[i])
			yArrayValues.extend(yvalues[idx]/(1* (10**7)))
			if i == 3196:
				break


		xarray = range(len(yArrayValues))
		plt.plot(xarray, yArrayValues)
		plt.title(audiofile)
		data_list.append(yArrayValues)
		if type-multi == True:
			if audioFileName == fileName2:
				temp2_train.append([1, 0])
			else:
				temp2_train.append([0, 1])
		else:
			if audioFileName == fileName2:
				temp2_train.append(1)
			else:
				temp2_train.append(0)
		x += 1
	return data_list, temp2_train
	
f = open(storeFile + '.pckl', 'wb')
pickle.dump(convert_train_data('E'), f)
f.close()

# Next 2 lines to show plot
# callFunc = convert_train_data('E')
# plt.show()