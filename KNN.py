from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import numpy.random 


def load_data():
    '''The method loads the data.'''
    mnist = fetch_openml("mnist_784")
    data = mnist["data"]
    labels = mnist["target"]  
    return (data, labels)


def generate_matrix(array):
    '''The method is a helper method for view_image method.'''
    matrix = np.zeros(shape=(28, 28))
    for i in range(27):
        matrix[i] = array[i*28:(i+1)*28]
    return matrix



def view_image(data):
    ''' The method plots the 256px image. 
        data = a 256 ints array, the same as the input data.'''
    matrix = generate_matrix(data)
    plt.imshow(matrix, cmap=plt.get_cmap('gray'))
    plt.show()

    
def divide_data(data, labels):
    ''' The method divides the data and its corresponding labels 
        into train and test.'''
    idx = numpy.random.RandomState(0).choice(70000,11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]] 
    return (train, train_labels, test, test_labels)
    

def calculate_distance(u, v):
    '''The method calculates the Euclidean distance between two vectors.'''
    dist = np.sqrt(np.sum((v-u)**2))
    return dist
    
 
def find_KNN_indices(v, k, train):
    ''' The method returns the indices of the knn in the train_set.'''
    dist_vector = np.full((1,len(train)), np.inf)
    for i in range(0,len(train)):
        dist_vector[0,i] = calculate_distance(train[i], v)
    indices_arr = dist_vector[0].argsort()[:k]
    return indices_arr



def KNN(train, train_labels, v, k):
    ''' The final method for q.1 - 
        # train - a set of images
        # train_labels - a vector of labels, corresponding to the images 
        # v - a query image
        # k - a number k
        
        returns - a prediction of the query image (digit)'''
    indices_arr = find_KNN_indices(v, k, train)
    digits = np.zeros((10,), dtype=int)
    for index in indices_arr:
        digit = int(train_labels[index])
        digits[digit] += 1
    prediction = digits.argsort()[-1:][0]
    return prediction



def compare(train, train_labels, test, test_labels, k, n):
    ''' The method checks the accuracy of the prediction 
        for all of the test data, 
        based on the k-nn and the first n images in the train data.
        
        returns P(test_label == prediction(test))'''
    correct_predictions = 0
    for index in range(0,len(test)):
        label = int(test_labels[index])
        prediction = int(KNN(train[:n], train_labels[:n], test[index], k))
        if label == prediction:
            correct_predictions += 1
    correct_prediction_pro = correct_predictions / len(test)
    return correct_prediction_pro

   
    
def k_variable_accuracy(train, train_labels, test, test_labels, max_k, n):
    ''' The method generates a vector of accuracies with respect to k.
        max_k - is the maximum k to test.
        
        returns a vector in length max_k 
        where vec[i] = P(test_label == prediction(test)| |test|=n & k = i)'''
    p_arr = np.zeros(max_k)
    for k in range(1, (max_k)+1):
        p = compare(train, train_labels, test, test_labels, k, n)
        p_arr[k-1] = p
    return p_arr


def plot_k_accuracy(y):
    ''' The method uses plt to plot the accuracy vector. '''    
    plt.plot(range(1,1+len(y)),y)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    

def n_variable_accuracy(train, train_labels, test, test_labels, 
                        k, min_n, max_n, gap_n):
    ''' The method generates a vector of accuracies with respect to n.
        max_n - is the maximum n to test.
        
        returns a vector in length max_n 
        where vec[i] = P(test_label == prediction(test)| |test|= i & k = k)'''
    n_arr = np.arange(min_n, (max_n)+1, gap_n)
    p_arr = np.zeros(len(n_arr))
    for n in range(0,len(n_arr)):
        p = compare(train, train_labels, test, test_labels, k, n)
        p_arr[n-1] = p
    return (n_arr, p_arr)


def plot_n_accuracy(x,y):
    ''' The method uses plt to plot the accuracy vector. '''    
    plt.plot(x,y)
    plt.xlabel("n")
    plt.ylabel("accuracy")
    


def main():
    (data, labels) = load_data()
    (train, train_labels, test, test_labels) = divide_data(data, labels)
    # q.2
    p = compare(train, train_labels, test, test_labels, 10, 1000)
    print("The accuracy of the prediction using the first 1000 training "+
          "images, on each of the test images using k = 10 is: " + str(p) 
          + ".")
    k_vec = k_variable_accuracy(train, train_labels, test, test_labels,10, 1000)
    # q.3
    plot_k_accuracy(k_vec)
    best_k = int(k_vec.argsort()[-1:][0])+ 1    # the accuracy of k is in y[k+1]
    print("The best k is " + str(best_k) + " .")
    """
    add discutions
    
    """

    # q.4
    n_vec = n_variable_accuracy(train, train_labels, test, test_labels,1, 
                            200,1000,100)
    plot_n_accuracy(n_vec)
    return n_vec


main()



