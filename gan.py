import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
    
Z = tf.placeholder(tf.float32, shape=[None, 100])

def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z, 128, activation=tf.nn.relu)
        x = tf.layers.dense(z, 784)
        x = tf.nn.sigmoid(x)
    return x
    

