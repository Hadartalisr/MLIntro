#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_model = svm.SVC(C=1000,kernel='linear')
    linear_model.fit(X_train, y_train)    
    linear_support_vectors = linear_model.n_support_
    create_plot(X_train, points[1], linear_model)
    plt.title('linear Classifier - # of support vectors is ' +
              str(sum(linear_support_vectors)) )
    # plt.show()
    
    rbf_model = svm.SVC(C=1000,kernel='rbf')
    rbf_model.fit(X_train, y_train)    
    rbf_support_vectors = rbf_model.n_support_
    create_plot(X_train, y_train, rbf_model)
    plt.title('rbf Classifier - # of support vectors is ' +
              str(sum(rbf_support_vectors)) )
    # plt.show()
    
    qudratic_model = svm.SVC(C=1000,kernel='poly',degree=2)
    qudratic_model.fit(X_train, y_train)    
    qudratic_support_vectors = qudratic_model.n_support_
    create_plot(X_train, y_train, qudratic_model)
    plt.title('qudratic Classifier - # of support vectors is ' +
              str(sum(qudratic_support_vectors)) )
    # plt.show()
    
    ret = np.array([linear_support_vectors, rbf_support_vectors, 
                    qudratic_support_vectors])
        
    return ret


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_vector = [pow(10,i) for i in range(-5,6)]
    train_accuracy_vector = []
    validation_accuracy_vector = []
    for C in C_vector:
        linear_model = svm.SVC(C=C,kernel='linear')
        linear_model.fit(X_train, y_train)    
        train_accuracy_vector.append(linear_model.score(X_train, y_train))
        validation_accuracy_vector.append(linear_model.score(X_val, y_val))
        
        if C in [pow(10,i) for i in range(-5,6,3)]:
            create_plot(X_train, y_train, linear_model)
            plt.title('linear Classifier - C = ' + str(C))
            plt.show()
        
    plt.plot(C_vector, train_accuracy_vector, label="train_accuracy")
    plt.plot(C_vector, validation_accuracy_vector, label="validation_accuracy")
    plt.xlabel("C")
    plt.legend()
    plt.ylabel("accuracy")
    plt.xscale("log")
    
    return validation_accuracy_vector
        
    

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    #gamma_vector = [pow(10,i) for i in range(-5,5)]
    gamma_vector = [i for i in range(1,15)]
    train_accuracy_vector = []
    validation_accuracy_vector = []
    for gammma in gamma_vector:
        rbf_model = svm.SVC(kernel='rbf',gamma=gammma)
        rbf_model.fit(X_train, y_train)    
        train_accuracy_vector.append(rbf_model.score(X_train, y_train))
        validation_accuracy_vector.append(rbf_model.score(X_val, y_val))
        """ create_plot """
        # create_plot(X_train, y_train, rbf_model)
        # plt.title('rbf Classifier - gamma = ' + str(gammma))
        # plt.show()
        
    plt.plot(gamma_vector, train_accuracy_vector, label="train_accuracy")
    plt.plot(gamma_vector, validation_accuracy_vector, label="validation_accuracy")
    plt.xlabel("gamma")
    plt.legend()
    plt.ylabel("accuracy")
    # plt.xscale("log")
    
    return validation_accuracy_vector
    
    
    
    

def text(data, labels):
    true_predictions = 0
    for i in range(0,len(data)):
        r = sign(np.dot(w, data[i]))
        if (r * labels[i]) > 0:
            true_predictions += 1
    accuracy = true_predictions/len(data)
    return accuracy    
    
points = get_points()
# ret = train_three_kernels(points[0], points[1], points[2], points[3])
# print(ret)
# ret2 = linear_accuracy_per_C(points[0], points[1], points[2], points[3])
# print(ret2)    
ret3 = rbf_accuracy_per_gamma(points[0], points[1], points[2], points[3])
print(ret3)  




