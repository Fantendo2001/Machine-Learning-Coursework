import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self, pseudocount=1):
        self.pseudocount = pseudocount
    
    def fit(self, X, y, numeric):
        self._class = y.unique()
        self._numeric_idx = list()
        for i, col in enumerate(X.columns):
            if numeric==[]:
                break
            if col in numeric:
                self._numeric_idx.append(i)
                numeric.remove(col)
        self._prior = dict()
        self._theta = list()
        for label in self._class:
            self._prior[label] = sum(y==label)/len(y)
            x = X[y==label]
            theta = [dict()for i in range(len(X.columns))]
            i = 0
            for idx in self._numeric_idx:
                while i < idx:
                    for attr in X.iloc[:,i].unique():
                        theta[i][attr] = (sum(x.iloc[:,i]==attr)+1)/(len(x)+len(X.iloc[:,i].unique()))
                    i += 1
                theta[i]['mu'] = x.iloc[:,i].mean()
                theta[i]['sigma'] = x.iloc[:,i].std()
                i += 1
            while i < len(X.columns):
                for attr in X.iloc[:,i].unique():
                    theta[i][attr] = (sum(x.iloc[:,i]==attr)+1)/(len(x)+len(X.iloc[:,i].unique()))
                i += 1
            self._theta.append(theta)  
    
    def predict(self,X):
        if len(self._numeric_idx)>0:
            from scipy.stats import norm
        pred = list()
        for row in range(len(X)):
            x = X.loc[row]
            Label = ''
            Posterior = 0
            for l, label in enumerate(self._class):
                posterior = self._prior[label]
                i = 0
                for idx in self._numeric_idx:
                    while i < idx:
                        posterior *= self._theta[l][i][x[i]]
                        i += 1
                    posterior *= norm.pdf(x[i],self._theta[l][i]['mu'],self._theta[l][i]['sigma'])
                    i += 1
                while i < len(X.columns):
                    posterior *= self._theta[l][i][x[i]]
                    i += 1
                if posterior > Posterior:
                    Posterior = posterior
                    Label = label
            pred.append(Label)
        pred = pd.Series(pred,name='predicted')     
        return pred

def cv(X, y, k=10, seed =None, numeric=['credit_amount','age','duration']):
    from tqdm import tqdm
    np.random.seed(seed)

    idx = np.arange(y.size) 
    np.random.shuffle(idx) 
    fold = np.array_split(idx,k) # split shuffled index into k folds
    
    pred = np.zeros_like(y)

    # cross validation using k folds
    for i in tqdm(range(k)):
        test_idx = fold[i]
        train_idx = np.setdiff1d(idx, test_idx)
        mod = NaiveBayes()
        mod.fit(X.loc[train_idx],y[train_idx],numeric=numeric.copy())
        pred[test_idx] = mod.predict(X.loc[test_idx].reset_index(drop=True))
    return pd.Series(pred,name='predicted') 

def evaluate(true,pred):
    cm = pd.crosstab(true,pred)
    TP = cm.iloc[1,1]
    TN = cm.iloc[0,0]
    FP = cm.iloc[0,1]
    FN = cm.iloc[1,0]
    precision = round(TP/ (TP+FP),3)
    sensitivity = round(TP / (TP + FN), 3)
    specificity = round(TN / (TN + FP), 3)
    F_measure = round(2*precision*sensitivity/(precision + sensitivity),3)
    print("\n=== Detailed Accuracy ===\n")
    print("Precision:", precision,sep='\t')
    print("Sensitivity:", sensitivity,sep='\t')
    print("Specificity:", specificity,sep='\t')
    print("F_measure:", F_measure,sep='\t')
    print("\n=== Confusion Matrix ===\n")
    print(cm)

def bagging(X,y,k,seed=None,numeric = ['credit_amount','age','duration']):
    from tqdm import tqdm
    np.random.seed(seed)
    pred = np.zeros_like(y)
    for i in tqdm(range(k)):
        idx = np.random.choice(len(X),len(X),replace=True)
        X_train = X.loc[idx].reset_index(drop=True)
        y_train = y.loc[idx].reset_index(drop=True)
        mod = NaiveBayes()
        mod.fit(X_train,y_train,numeric=numeric.copy())
        pred += (mod.predict(X)=="good")
    pred = pd.Series(np.where(pred > k//2,'good','bad'),name='predicted')
    return pred
