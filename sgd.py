#################################
# Your name: Hadar Tal
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

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
    w = np.zeros(784)
    for t in range(1,T+1) :
        eta = (eta_0/t)
        i = np.random.randint(len(data))
        x = data[i]
        y = labels[i] # The data labeling is already in +-1 form (8 = 1, 0 = -1) 
        r = y * np.dot(x,w) # prediction
        w = (1-eta)*w 
        if r < 1:        
            w = np.add(w, eta * C * y * x)
    return w 

        



def SGD_ce(data, labels, eta_0, T):
    classifiers = np.zeros((10,784))
    for t in range(1,T+1) :
        i = np.random.randint(len(data))
        x = data[i]
        y = labels[i] # The data labeling conatins all the decimal digits
        gradients = calculate_gradients(classifiers,x,y)
        gradients = [(-1*eta_0) * gradients[i] for i in range(10)]
        classifiers = np.add(classifiers, gradients)
    return classifiers
    
    
    
	
#################################



#################################


def view_image(data):
    ''' The method plots the 784px image. 
        data = a 784 ints array, the same as the input data.'''
    matrix = generate_matrix(data)
    plt.imshow(matrix, cmap = 'viridis', interpolation='nearest')
    plt.show()


def sign(r): 
    return (1 if r>0 else -1)


def SGD_hinge_test(w, data, labels):
    true_predictions = 0
    for i in range(0,len(data)):
        r = sign(np.dot(w, data[i]))
        if (r * labels[i]) > 0:
            true_predictions += 1
    accuracy = true_predictions/len(data)
    return accuracy


def find_best_eta(train_data, train_labels, validation_data, validation_labels, C, T):    
    etas = [np.float_power(10, -1 + (0.01*k)) for k in range(0,100)]
    # for the first observation - etas = [np.float_power(10, k) for k in range(-4,4)]
    avg_accuracy = []
    for eta in etas:
        accuracy_v = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            accuracy = SGD_hinge_test(w, validation_data, validation_labels)
            accuracy_v.append(accuracy)
        avg_accuracy.append(np.average(accuracy_v))
    plt.plot(etas, avg_accuracy)
    plt.xlabel("eta")
    plt.ylabel("average accuracy")
    plt.xscale("log")
    u = np.argsort(avg_accuracy)[-1:][0]
    return etas[u]


def find_best_C(train_data, train_labels, validation_data, validation_labels, T, eta):    
    cs = [np.float_power(10, -0.4 + (0.01*k)) for k in range(-100, 100)]
    # for the first observation - 
    # cs = [np.float_power(10, 0.1*k) for k in range(-100,-10)]
    avg_accuracy = []
    for c in cs:
        accuracy_v = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta, T)
            accuracy = SGD_hinge_test(w, validation_data, validation_labels)
            accuracy_v.append(accuracy)
        avg_accuracy.append(np.average(accuracy_v))
    plt.plot(cs, avg_accuracy)
    plt.xlabel("C")
    plt.ylabel("average accuracy")
    plt.xscale("log")
    u = np.argsort(avg_accuracy)[-1:][0]
    print(avg_accuracy[u]) 
    return cs[u]



def q3(train_data, train_labels,test_data, test_labels):
    C = 0.20892961
    eta = 0.01148154
    T = 2000
    w = SGD_hinge(train_data, train_labels, C, eta, T)
    view_image(w)


    
def calculate_gradients(classifiers,x,y):
    probs = calculate_probabilities(classifiers,x)
    probs[y] -= 1
    gradients = [probs[i] * x for i in range(10)]    
    return gradients


def calculate_probabilities(classifiers,x):
    dots = [np.dot(x, classifiers[i]) for i in range(10)] 
    max_dot = np.max(dots)
    dots = dots - max_dot # the exponent for the real dot is too high
    # print(dots)
    e_to_dot = np.exp(dots)
    probs = e_to_dot / np.sum(e_to_dot)
    # print(probs)
    return probs
    

def generate_matrix(array):
    '''The method is a helper method for view_image method.'''
    matrix = np.zeros(shape=(28, 28))
    for i in range(27):
        matrix[i] = array[i*28:(i+1)*28]
    return matrix


def SGD_ce_test(classifiers, data, labels):
    true_predictions = 0
    for i in range(0,len(data)):
        p_v = [np.dot(classifiers[d],data[i]) for d in range(10)]
        y_hat = np.argsort(p_v)[-1:][0]
        if labels[i] == y_hat :
            true_predictions += 1
    accuracy = true_predictions/len(data)
    return accuracy


def ce_find_best_eta(train_data, train_labels, validation_data, validation_labels, T):    
    etas = [np.float_power(10, -7 + (0.02*k)) for k in range(0,50)]
    # for the first observation - etas = [np.float_power(10, k) for k in range(-10,10)]
    avg_accuracy = []
    for eta in etas:
        accuracy_v = []
        for i in range(10):
            classifiers = SGD_ce(train_data, train_labels, eta, T)
            accuracy = SGD_ce_test(classifiers, validation_data, validation_labels)
            accuracy_v.append(accuracy)
        avg_accuracy.append(np.average(accuracy_v))
    plt.plot(etas, avg_accuracy)
    plt.xlabel("eta")
    plt.ylabel("average accuracy")
    plt.xscale("log")
    u = np.argsort(avg_accuracy)[-1:][0]
    return ce[u]


def ce_plot_weights(test_data, test_labels, T):    
    eta = 6.30957344e-07
    classifiers = SGD_ce(test_data, test_labels, eta, T)
    num = 0
    row = 0
    fig, axs = plt.subplots(2, 5)
    j = 0
    while(j < 10):
        matrix = generate_matrix(classifiers[j])
        axs[row,num].imshow(matrix, cmap = 'viridis', interpolation='nearest')
        num += 1
        j += 1
        if num == 5:
            num = 0
            row = 1
            
            
def ce_q3(train_data, train_labels, test_data, test_labels, T, eta):    
    classifiers = SGD_ce(train_data, train_labels, eta, T)
    accuracy = SGD_ce_test(classifiers, test_data, test_labels)
    print(accuracy)



def main():
    # q1
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    # q1.a
    best_eta = find_best_eta(train_data, train_labels, validation_data, validation_labels, 1, 1000)
    print(best_eta)
    # q1.b
    best_c = find_best_C(train_data, train_labels, validation_data, validation_labels, 1000, best_eta)
    print(best_c)
    # q1.c
    w = SGD_hinge(train_data, train_labels, best_c, best_eta,20000)
    view_image(w)
    # q1.d
    accuracy_v = []
    for i in range(10):
        w = SGD_hinge(train_data, train_labels, best_c, best_eta, 20000)
        accuracy = SGD_hinge_test(w, test_data, test_labels)
        accuracy_v.append(accuracy)
    accuracy = np.average(accuracy_v)
    print(accuracy)
    
    # q2    
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    train_labels = np.array(train_labels, dtype=int) 
    validation_labels = np.array(validation_labels, dtype=int)  
    test_labels = np.array(test_labels, dtype=int)  
    # q2.a
    best_eta = ce_find_best_eta(train_data, train_labels, validation_data, validation_labels, 1, 1000)
    print(best_eta)
    # q2.b
    ce_plot_weights(test_data, test_labels, 20000)
    #q2.c
    q3(train_data, train_labels, test_data, test_labels, 2000, best_eta)

    
main()


