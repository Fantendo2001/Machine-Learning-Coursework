import pandas as pd 
import numpy as np

class NaiveBayes:
    def __init__(self, pseudo = 1):
        self.pseudo = pseudo
        
    def _binarize(self, X):
        return (X>0).astype(np.int32)
        
    def fit(self, X, y):
        X = self._binarize(X)
        n, k = X.shape
        C = len(np.unique(y)) # number of classes
        self.theta = np.zeros([C,k]) # mean of each feature per class
        self.prior = np.zeros(C)
        for c in range(C):
            self.theta[c] = (X[y==c].sum()+self.pseudo) / (sum(y==c)+self.pseudo*C)
            self.prior[c] = sum(y==c) / n
    
    def predict(self, X):
        X = self._binarize(X)
        log_prior = np.log(self.prior)
        log_post = X@np.log(self.theta.T) + (1-X)@np.log(1-self.theta.T) + log_prior
        return log_post.idxmax(axis=1)
    
    def predict_error(self, X, y):
        return sum(self.predict(X) != y)/y.size