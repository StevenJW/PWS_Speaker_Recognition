import os
import re
import sys
import wave

import numpy
import librosa
import tflearn
import skimage.io
import numpy as np
import tensorflow as tf

class Source:  # labels
	DIGIT_WAVES = './spoken_numbers_pcm.tar'
	# DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
	DIGIT_SPECTROS = './spoken_numbers_spectros_64x64.tar'
	NUMBER_WAVES = 'spoken_numbers_wav.tar'
	NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
	WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width,height=512# todo: sliding window!
	SPEECH_COMMANDS = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz" # 16000Hz 1ch s16le (2 bytes per sample)
	WORD_WAVES = 'spoken_words_wav.tar'
	TEST_INDEX = 'test_index.txt'
	TRAIN_INDEX = 'train_index.txt'

def load_wav_file(name):
	f = wave.open(name, "rb")
	# print("loading %s"%name)
	chunk = []
	data0 = f.readframes(CHUNK)
	while data0:  # f.getnframes()
		# data=numpy.fromstring(data0, dtype='float32')
		# data = numpy.fromstring(data0, dtype='uint16')
		data = numpy.fromstring(data0, dtype='uint8')
		data = (data + 128) / 255.  # 0-1 for Better convergence
		# chunks.append(data)
		chunk.extend(data)
		data0 = f.readframes(CHUNK)
	# finally trim:
	chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
	chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
	# print("%s loaded"%name)
	return chunk

def get_speakers(path=pcm_path):
	maybe_download(Source.DIGIT_SPECTROS)
	maybe_download(Source.DIGIT_WAVES)
	files = os.listdir(path)
	def nobad(name):
		return "_" in name and not "." in name.split("_")[1]
	speakers=list(set(map(speaker,filter(nobad,files))))
	print(len(speakers)," speakers: ",speakers)
	return speakers

def maybe_download(file, work_directory=DATA_DIR):
	"""Download the data from Pannous's website, unless it's already here."""
	print("Looking for data %s in %s"%(file,work_directory))
	if not os.path.exists(work_directory):
		try:
			os.mkdir(work_directory)
		except:
			pass
	filepath = os.path.join(work_directory, re.sub('.*\/','',file))
	if not os.path.exists(filepath):
		if not file.startswith("http"): url_filename = SOURCE_URL + file
		else: url_filename=file
		print('Downloading from %s to %s' % (url_filename, filepath))
		filepath, _ = urllib.request.urlretrieve(url_filename, filepath,progresshook)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
		# os.system('ln -s '+work_directory)
	if os.path.exists(filepath):
		print('Extracting %s to %s' % ( filepath, work_directory))
		os.system('tar xf '+filepath+" -C "+work_directory)
		print('Data ready!')
	return filepath.replace(".tar","")

def wave_batch_generator(batch_size=10,source=Source.DIGIT_WAVES,target=Target.digits): #speaker
	maybe_download(source, DATA_DIR)
	if target == Target.speaker: speakers=get_speakers()
	batch_waves = []
	labels = []
	# input_width=CHUNK*6 # wow, big!!
	files = os.listdir(path)
	while True:
		shuffle(files)
		print("loaded batch of %d files" % len(files))
		for wav in files:
			if not wav.endswith(".wav"):continue
			if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
			elif target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
			elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
			else: raise Exception("todo : Target.word label!")
			chunk = load_wav_file(path+wav)
			batch_waves.append(chunk)
			# batch_waves.append(chunks[input_width])
			if len(batch_waves) >= batch_size:
				yield batch_waves, labels
				batch_waves = []  # Reset for next batch
				labels = []

def one_hot_to_item(hot, items):
	i=np.argmax(hot)
	item=items[i]
	return item


#Actual thing

# Simple speaker recognition demo, with 99% accuracy in under a minute ( on digits sample )

# | Adam | epoch: 030 | loss: 0.05330 - acc: 0.9966 -- iter: 0000/1000
# 'predicted speaker for 9_Vicki_260 : result = ', 'Vicki'

print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() # why? works on Mac?

speakers = get_speakers()
number_classes=len(speakers)
print("speakers",speakers)

batch=wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
X,Y=next(batch)


# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)

# demo_file = "8_Vicki_260.wav"
demo_file = "8_Bruce_260.wav"
demo=data.load_wav_file(data.path + demo_file)
result=model.predict([demo])
result=data.one_hot_to_item(result,speakers)
print("predicted speaker for %s : result = %s "%(demo_file,result)) # ~ 97% correct