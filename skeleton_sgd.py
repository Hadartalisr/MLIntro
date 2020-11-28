#################################
# Your name: Hadar Tal
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import math
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_hinge():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']

	neg, pos = "0", "8"
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = (labels[train_idx[:6000]] == pos)*2-1

	validation_data_unscaled = data[train_idx[6000:], :].astype(float)
	validation_labels = (labels[train_idx[6000:]] == pos)*2-1

	test_data_unscaled = data[60000+test_idx, :].astype(float)
	test_labels = (labels[60000+test_idx] == pos)*2-1

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def helper_ce():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']
	
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = labels[train_idx[:6000]]

	validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
	validation_labels = labels[train_idx[6000:8000]]

	test_data_unscaled = data[8000+test_idx, :].astype(float)
	test_labels = labels[8000+test_idx]

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    w = init_start_weights(np.ndarray.min(data), np.ndarray.max(data))
    for t in range(1,T+1) :
        eta = eta_0/t
        i = np.random.randint(len(data))
        x = data[i]
        y = labels[i] # The data labeling is already in +-1 form (8 = 1, 0 = -1) 
        r = y * np.dot(x,w) # prediction
        w = np.multiply((1-eta_0),w) 
        if r < 1:        
            w = np.add(w, eta * C * y * x)
    return w 

        



def SGD_ce(data, labels, eta_0, T):
	"""
	Implements multi-class cross entropy loss using SGD.
	"""
	# TODO: Implement me
	pass
	
#################################



#################################

def generate_matrix(array):
    '''The method is a helper method for view_image method.'''
    matrix = np.zeros(shape=(28, 28))
    for i in range(27):
        matrix[i] = array[i*28:(i+1)*28]
    return matrix


def init_start_weights(low,high):
    return np.random.uniform(low=low,high=high, size=(784,))


def view_image(data):
    ''' The method plots the 784px image. 
        data = a 784 ints array, the same as the input data.'''
    matrix = generate_matrix(data)
    plt.imshow(matrix, cmap=plt.get_cmap('gray'))
    plt.show()


def sign(r): 
    return (1 if r>0 else -1)


def SGD_hinge_test(w, test_data, test_labels):
    false_predictions = 0
    for i in range(0,len(test_data)):
        r = sign(np.dot(w, test_data[i]))
        if r * test_labels[i] < 0:
            false_predictions += 1
    accuracy.append((false_predictions/len(test_data)))
    return accuracy


def find_best_eta(data, labels, C, eta_0, T):
    







train_data, train_labels, validation_data, validation_labels, test_data, test_labels = skeleton_sgd.helper_hinge()

w = SGD_hinge(train_data, train_labels, 1, pow(10,-5) , 1000)


w = SGD_hinge(train_data, train_labels, 1, 1 , 1000)










