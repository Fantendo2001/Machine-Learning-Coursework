import numpy as np
import matplotlib.pyplot as plt

class Dense():
    def __init__(self, in_dim, out_dim, activation):
        self.w = np.random.normal(loc=0.0, 
                                  scale = np.sqrt(2/(in_dim+out_dim)),
                                  size = (in_dim, out_dim))
        self.b= np.zeros(out_dim)
        self.activation = activation
        
    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(-x))
        if self.activation == 'tanh':
            return np.tanh(x)
        if self.activation == 'relu':
            return np.maximum(0,x)
        return x
    
    def _gradient(self,x):
        if self.activation == 'sigmoid':
            x = self._activate(x)
            return x*(1-x)
        if self.activation == 'tanh':
            x = self._activate(x)
            return 1-x**2
        if self.activation == 'relu':
            x = self._activate(x)                      
            return np.sign(x)
        return np.ones_like(x)
    
    def forward(self,inp):
        return self._activate(np.dot(inp,self.w) + self.b)
    
    def backward(self,inp,grad_out,lr,regularization,lmbd):
        delta = self._gradient(inp@self.w)*grad_out
        grad_inp = delta@self.w.T
        if regularization == 'L2':
            self.w -= lr * lmbd / len(inp) * self.w
        elif regularization == 'L1':
            self.w -= lr * lmbd / len(inp) * np.sign(self.w)
        self.w = self.w - lr/ len(inp) * inp.T@delta
        self.b = self.b - lr/ len(inp) * delta.mean(axis=0)*inp.shape[0]
        
        return grad_inp

class NeuralNetwork:
    def __init__(self):
        self._layer = list()
    
    def add_layer(self, in_dim, out_dim, activation):
        self._layer.append(Dense(in_dim, out_dim, activation))
    
    def _loss(self,logit,y):
        if self.cost == 'CrossEntropy':
            fx = logit[np.arange(len(logit)),y]
            return np.mean(np.log(np.sum(np.exp(logit),axis=-1))-fx)
        if self.cost == 'MSE':
            enc = np.zeros_like(logit)
            enc[np.arange(len(logit)),y] = 1
            softmax = np.exp(logit) / np.exp(logit).sum(axis=1,keepdims=True)
            return np.mean((softmax-enc)**2)
    
    def _grad(self,logit,y):        
        if self.cost == 'CrossEntropy':
            enc = np.zeros_like(logit)
            enc[np.arange(len(logit)),y] = 1
            softmax = np.exp(logit)/ np.exp(logit).sum(axis=-1,keepdims=True)
            return (- enc + softmax) / logit.shape[0]
        if self.cost == 'MSE':
            fx = np.vstack(logit[np.arange(len(logit)),y])
            enc = np.zeros_like(logit)
            enc[np.arange(len(logit)),y] = 1
            softmax = np.exp(logit) / np.exp(logit).sum(axis=1,keepdims=True)
            return (softmax**2 - softmax*(enc-fx+softmax.sum(axis=1,keepdims=True)))*2/logit.size

    def _feedfwd(self,X):
        layer_inp = []
        inp = X
        for layer in self._layer:
            layer_inp.append(inp)
            inp = layer.forward(inp)
        return layer_inp, inp
    
    def _bwdprop(self, layer_inp, grad_out):
        l = len(self._layer)-1
        while l>=0:
            layer = self._layer[l]
            grad_out = layer.backward(layer_inp[l],grad_out,self.lr,self.regularization,self.lmbd) 
            l -= 1
        
    def _iter_batch(self, X, y, batch):
        from tqdm import trange
        indices = np.random.permutation(len(X))
        for i in trange(0, len(X) - batch + 1, batch):
            index = indices[i:i + batch]
            yield X[index], y[index]
    
    def _train(self,X,y):
        layer_inp, logit = self._feedfwd(X)
        grad_out = self._grad(logit,y)
        self._bwdprop(layer_inp,grad_out)
        return (self._loss(logit,y))
    
    def fit(self,X_train,y_train,X_valid,y_valid,epoch,batch,lr,cost,regularization=None,lmbd=0,show=False):
        from IPython.display import display, clear_output
        
        self.lr = lr 
        self.cost = cost
        self.regularization = regularization
        self.lmbd = lmbd
        
        train_log = []
        valid_log = []
        for i in range(epoch):
            for x_batch,y_batch in self._iter_batch(X_train,y_train,batch):
                self._train(x_batch,y_batch)
            train_log.append(np.mean(self.predict(X_train)==y_train))
            valid_log.append(np.mean(self.predict(X_valid)==y_valid))
            clear_output(wait=True)
            display("Epoch:"+str(i+1)+", "+\
                    "Train accuracy:"+str(train_log[-1])+", "+\
                    "Valid accuracy:"+str(valid_log[-1]))
            if show:
                plt.plot(train_log,label='train accuracy')
                plt.plot(valid_log,label='valid accuracy',linestyle='--')
                plt.legend(loc='best')
                plt.grid()
                plt.show()
                
        return {'train':train_log, 'valid':valid_log}
    
    def predict(self,X):
        logit = self._feedfwd(X)[-1]
        return logit.argmax(axis=-1)
