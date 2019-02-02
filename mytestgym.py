import tensorflow as tf
import gym
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32,shape=[None,num_inputs])

hidden_layer = tf.layers.dense(X,num_hidden,activation=tf.nn.elu,kernel_initializer=initializer)

logits = tf.layers.dense(hidden_layer,num_outputs)
outputs = tf.nn.sigmoid(logits)
