import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa
import librosa.display
from scipy.io import wavfile as wav

audiofile = 'AudioFiles/Ematthijs1.wav'

x, fs = librosa.load(audiofile)
mfccs = librosa.feature.mfcc(x, sr=fs)

plt.title(audiofile)
plt.plot(mfccs)
plt.show()

'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.tight_layout()
plt.show()
'''