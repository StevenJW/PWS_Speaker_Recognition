from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from pathlib import Path

import librosa
#import librosa.display

import numpy as np



#plt.show()

colors = False

z = 1
while z <= 1:
	audiofile = '../AudioFiles/Esteven' + str(z) + '.wav'
	if Path(audiofile).is_file():
		x, fs = librosa.load(audiofile)
		mfccs = librosa.feature.mfcc(x, sr=fs)
		print(mfccs.shape)
		print(mfccs[0])
		print(mfccs[1])
		print(mfccs[2])
		print(mfccs[3])

		if colors:
			plt.figure(figsize=(10, 4))
			librosa.display.specshow(mfccs, x_axis='time')
			plt.colorbar()
			plt.tight_layout()
		else:
			plt.title(audiofile)
			plt.plot(mfccs)
	else:
		break
	z += 1

plt.show()