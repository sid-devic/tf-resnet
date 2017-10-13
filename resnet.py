import tensorflow as tf
import numpy as np

def summary(x):
	'''
	Provides a summary node, implement where needed for a quick visualization of the data at that point
	'''

	tensorName = x.op.name 
	tf.summary.histogram(tensorName + '/activations', x) # Visualizes distribution of data in X
	tf.summary.scalar(tensorName + '/sparsity', tf.nn.zero_fraction(x)) # Returns a scalar corresponding to the fraction of X's elements that are zeros (sparsity)

def tensorboard_graph(train_dir='logs'):
	'''	
	Display computational graph structure on tensorboard
	'''
	inputTensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32) # Creates a 4D tensor filled with ones (using numpy). Data type 32bit float.
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
