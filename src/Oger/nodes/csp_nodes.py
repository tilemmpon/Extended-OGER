import numpy as np
import scipy as sp
import mdp

class CSPNode(mdp.Node):
    
    def __init__(self, outputs=None, input_dim=None, output_dim=None, dtype=None):
        '''
        outputs sets the list of outputs after applying csp for example outputs=[0,1,2,-3,-2,-1]
        '''
        if outputs != None:
            output_dim = len(outputs)
        super(CSPNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._Rp, self._Rn = None, None
        self._outputs = outputs
    
    def _train(self,x,y):
        if self._Rp == None:
            self._Rp = np.zeros((x.shape[1], x.shape[1]))
            self._Rn = np.zeros((x.shape[1], x.shape[1]))
            if self._outputs == None:
                self._outputs = range(x.shape[1])
        y = y>0.0001
        p = np.nonzero(y)[0]
        n = np.nonzero(1-y)[0]
        self._Rp += np.dot(x[p,:].T, x[p,:])
        self._Rn += np.dot(x[n,:].T, x[n,:])
    
    def _stop_training(self):
        self._Rp /= np.trace(self._Rp)
        self._Rn /= np.trace(self._Rn)
        R = self._Rp + self._Rn
        
        D, C = np.linalg.eigh(R)
        i = np.argsort(D)
        D = D[i]
        C = C[:,i]
        
        if np.sum(D<0)>0:
            D=np.abs(D)
            print 'A numerical error occurred at a csp node, but it should be safe to continue...'
        
        # Whitening
        W = np.dot(np.diag(1. / np.sqrt(D)), C.T)
        
        # Generalized eigenvalue
        Sp = np.dot(W, np.dot(self._Rp, W.T))
        Sn = np.dot(W, np.dot(self._Rn, W.T))
        D, C = sp.linalg.eig(a=Sp, b=Sp+Sn)
        D, C = np.real(D), np.real(C)
        i = np.argsort(D)[::-1]
        D = D[i]
        C = C[:,i]
        
        self.CSP = np.dot(C.T, W).T
        self.eig = D
        self.W = self.CSP[:, self._outputs]
    
    def _execute(self, x):
        return np.dot(x, self.W)