import pandas as pd 
import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, λ, lr = 1,alpha=0.5, beta=0.5, max_iter=10000, tol = 1e-3, 
                 add_intercept = True, transform = None):
        self.lr = lr # learning rate
        self.alpha = alpha # step size
        self.beta = beta # step size shrinking factor 
        self.max_iter = max_iter # maximal interations
        self.λ = λ # regularization term
        self.iters = 0 # number of iterations
        self.tol = tol # stopping criteria of backtracking line search
        self.add_intercept = add_intercept # add intercept to X by default 
        self.transform = transform # type of transformation: stnd, log or binary
        
    def _transform(self, X):
        if self.transform == "stnd": # standardize
            X = (X-X.mean())/X.std()
        elif self.transform == "log": # logarithmize
            X = np.log(X+0.1)
        elif self.transform == "binary": #binarize
            X = (X>0).astype(np.int32)
        if self.add_intercept:
            X = np.hstack([np.ones((X.shape[0],1)), X])
        return X
    
    def _sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
    
    def _loss(self, y, μ, w):
        return (sum(-np.log(μ[y==1]))+sum(-np.log(1-μ[y==0])) + self.λ*sum(w**2)) / y.size
    
    def _gradient(self, X, y, μ, w):
        return (X.T@(μ-y)) / y.size + 2*self.λ*w / y.size
        
    def fit(self, X, y):
        self.converged = False
        
        X = self._transform(X)
        n, k = X.shape
        
        '''
        gradient descent with backtracking line search
        '''
        prev_w = self.w = np.zeros(k)
        μ = self._sigmoid(X@self.w)
        prev_loss = self.loss = self._loss(y, μ, self.w)
        gradient = self._gradient(X, y, μ, self.w)
        
        for i in range(self.max_iter):
            norm_gradient = np.sqrt(sum(gradient**2))
            if (norm_gradient < self.tol):
                self.converged = True
                break  
            
            self.iters += 1
            t = self.lr
            self.w = prev_w - t * gradient
            μ = self._sigmoid(X@self.w)                       
            self.loss = self._loss(y, μ, self.w)
            
            while (self.loss > prev_loss - self.alpha*t*norm_gradient**2):
                t = t*self.beta
                self.w = prev_w - t*gradient
                μ = self._sigmoid(X@self.w)
                self.loss = self._loss(y, μ, self.w) 
                
            prev_w = self.w
            prev_loss = self.loss
            gradient = self._gradient(X, y, μ, self.w)
        
    def predict_prob(self, X):
        X = self._transform(X)
        return self._sigmoid(X@self.w)
        
    def predict(self, X):
        return (self.predict_prob(X)>0.5).astype(np.int32)
    
    def predict_error(self,X,y):
        return sum(self.predict(X) != y)/y.size

'''
lambda parameter tuning using cross validation
'''
def CV(X, y, lambdas = 10**np.linspace(-2,1,10), transform = None, k = 10, seed = None):
    np.random.seed(seed)
    errors = np.zeros_like(lambdas)
    
    idx = np.arange(y.size) 
    np.random.shuffle(idx) 
    fold = np.array_split(idx,k) # split shuffled index into k folds

    # cross validation using k folds
    for i in tqdm(range(k)):
        for l, lam in enumerate(lambdas):
            test_idx = fold[i]
            train_idx = np.setdiff1d(idx, test_idx)
            mod = LogisticRegression(lam,transform = transform)
            mod.fit(X.loc[train_idx],y[train_idx])
            errors[l] += mod.predict_error(X.loc[test_idx],y[test_idx])
    errors /= k
    Lam = lambdas[np.argmin(errors)]
    return Lam