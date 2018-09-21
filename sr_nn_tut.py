from __future__ import print_function
import tensorflow as tf

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=true)

# hyperparameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# network parameters
# 28 x 28 image
n_input = 784
# 10 digits
n_classes = 10
# A probability that certain neurons will be turned off so that new pathways can be made and the NN doesn't become static
dropout = 0.75

# Placeholder for image so that it flows into the NN
x = tf.placeholder(tf.float32, [None, n_input])
# Placeholder for the 10 different classes, so the label an image can have
y = tf.placeholder(tf.float32, [None, n_classes])
# Placeholder for the dropout, so that certain neurons get turned of now and then.
keep_prob = tf.placeholder(tf.float32)

# Convulutional layers create like a filter over the image, so basically they convert it so it can be put in the neural net.
# AKA Pre-processing :)
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

# The pool layers creates 'pools' of data from the image and gives it one output
def maxpool2d(x, k=2)
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding = 'SAME')


