import numpy as np
from scipy.linalg import norm

class EMalgo:
    def __init__(self):
        pass
    
    def _EM(self):
        mu = np.zeros(self.d) 
        sigma = np.zeros([self.d,self.d])

        for i, row in enumerate(self.data):
            u = self._missing[i,:]
            o = ~u
            x = row[o]
            v = self._sigma[u,:][:,u]-self._sigma[u,:][:,o]@np.linalg.pinv(self._sigma[o,:][:,o])@self._sigma[o,:][:,u]
            m = self._mu[u]+self._sigma[u,:][:,o]@np.linalg.inv(self._sigma[o,:][:,o])@(x-self._mu[o])
            
            Ex = np.zeros(self.d) 
            Ex[u] = m
            Ex[o] = x
            
            Exx =np.zeros([self.d,self.d])
            Exx[np.outer(u,u)] = (np.outer(m,m)+v).flatten()
            Exx[np.outer(o,o)] = np.outer(x,x).flatten()
            Exx[np.outer(o,u)] = np.outer(x,m).flatten()
            Exx[np.outer(u,o)] = np.outer(m,x).flatten()
            
            mu += Ex
            sigma += Exx
            
        mu = mu/self.n
        sigma = sigma/self.n-np.outer(mu,mu)
        return mu, sigma
                        
    def fit(self,data,maxiter = 1000,delta=1e-05):
        from tqdm import tqdm
        from IPython.display import display, clear_output
        self.data = data
        self.n, self.d = data.shape
        self._missing = np.isnan(data)
        complete = data[~self._missing.any(axis=1)]
        if (len(complete)>0):
            self._mu = np.mean(complete,axis=0)
            self._sigma = np.cov(complete.T)
        else:
            filled = np.nan_to_num(data)+self._missing*np.random.standard_normal(data.shape)
            self._mu = np.mean(filled,axis=0)
            self._sigma = np.cov(filled.T)
        
        for i in range(maxiter):
            mu, sigma = self._EM()
            error_mu = norm(mu-self._mu)/norm(mu)
            error_sigma = norm(sigma-self._sigma)/norm(sigma)
            if (error_mu < delta) and (error_sigma < delta):
                print("iters:",i,"error_mu:",error_mu,"error_sigma:",error_sigma)
                return error_mu,error_sigma
            self._mu = mu
            self._sigma = sigma