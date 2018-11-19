import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa
import librosa.display
from scipy.io import wavfile as wav
from pathlib import Path


#plt.title(audiofile)
#plt.plot(mfccs)
#print(mfccs)
#plt.show()


z = 1
while z <= 4:
	audiofile = 'AudioFiles/Esteven' + str(z) + '.wav'
	if Path(audiofile).is_file():
		x, fs = librosa.load(audiofile)
		mfccs = librosa.feature.mfcc(x, sr=fs)
		plt.figure(figsize=(10, 4))
		librosa.display.specshow(mfccs, x_axis='time')
		plt.colorbar()
		plt.tight_layout()
	else:
		break
	z += 1

plt.show()