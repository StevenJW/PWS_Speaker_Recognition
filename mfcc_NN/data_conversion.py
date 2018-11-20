import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from pathlib import Path
import numpy as np

import librosa
import pickle
import wave
import sys


def convert_train_data(audioVowel):
	global fileName1
	global fileName2

	# Train Data
	print('Train data')
	temp_train = []
	list_train = []
	print('Steven train data')
	list_train, y_data = convert_train_data_from(fileName1, audioVowel, list_train)
	temp_train.extend(y_data)
	print('Matthijs train data')
	list_train, y_data = convert_train_data_from(fileName2, audioVowel, list_train)
	temp_train.extend(y_data)

	# Test data
	print('Test data')
	temp_train_test = []
	list_train_test = []
	list_train_test, y_data = convert_train_data_from(fileName1 + 'Test', audioVowel, list_train_test)
	temp_train_test.extend(y_data)

	list_train_test, y_data = convert_train_data_from(fileName2 + 'Test', audioVowel, list_train_test)
	temp_train_test.extend(y_data)

	return np.array(list_train), np.array(temp_train), np.array(list_train_test), np.array(temp_train_test)
	
def convert_train_data_from(audioFileName, vowel, data_list):
	global fileName2
	b = 1
	temp2_train = []
	while True:
		#if b > 25:
		#	break

		audiofile = '../AudioFiles/'+ str(vowel) + str(audioFileName) + str(b) + '.wav'
		if Path(audiofile).is_file():
			spf = wave.open(audiofile,'r')
		else: 
			break

		print(b)
		
		x, fs = librosa.load(audiofile)
		mfccs = librosa.feature.mfcc(x, sr=fs)
		print(mfccs.shape)

		#plt.plot(xarray, yArrayValues)
		#plt.title(audiofile)
		data_list.append(mfccs[19][0:120])
		if typeMulti == True:
			if audioFileName == fileName2:
				temp2_train.append([1, 0])
			else:
				temp2_train.append([0, 1])
		else:
			if audioFileName == fileName2:
				temp2_train.append(1)
			else:
				temp2_train.append(0)
		b += 1
	return data_list, temp2_train
	

fileName1 = 'steven'
fileName2 = 'matthijs'
typeMulti = True
testFile = False

z = 1
if typeMulti:
	storeFile = 'multi_'
	while True:
		if not Path(storeFile + str(z) + '.pckl').is_file():
			storeFile = storeFile + str(z)
			break
		z += 1

f = open(storeFile + '.pckl', 'wb')
pickle.dump(convert_train_data('E'), f)
f.close()


# Next 2 lines to show plot
# callFunc = convert_train_data('E')
# plt.show()