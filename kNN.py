import numpy as np
from collections import Counter

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
    # training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. Calculate the distance betweet y and each 
    #               row of X, find  the k closest observations and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance between two vectors A and B use
    #       use the np.linalg.norm() function.
    
    dist_kNN = [np.linalg.norm(y-x) for x in X[:k,:]] 
    idx_kNN = [i for i in range(k)]
    
    ZIP = sorted(zip(dist_kNN,idx_kNN))
    
    dist_kNN = [d for d,i in ZIP]
    idx_kNN = [i for d,i in ZIP]
    
    idx = k
    for x in X[k:,:]:
        d = np.linalg.norm(y-x)
        counter = k-1
        
        while d < dist_kNN[counter] and counter >=0:
            counter -= 1
        
        dist_kNN.insert(counter+1, d)
        dist_kNN.pop()
        
        idx_kNN.insert(counter+1, idx)
        idx_kNN.pop()
        
        idx += 1
    
    
    # return the label of the test data
    labels_kNN = [labels[i] for i in idx_kNN]
     
    label = most_common(labels_kNN)
    
    return label