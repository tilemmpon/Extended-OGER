import mdp
import Oger
import numpy as np
import scipy as sp
import scipy.spatial.distance as dist

class ELMNode(mdp.Node):
    """
    Extreme Learning Machine Node

    Implements code from the OP-ELM toolbox in Python. For OP-ELM behaviour use OPRidgeRegressionNode(ridge_param=0) for training.
    """

    def __init__(self, output_dim=100, use_linear=True, use_sigmoid=True, use_gaussian=True, zero_mean=True, unit_var=True, input_dim=None, dtype='float64'):
        """ Initializes and constructs a random ELM.
        Parameters are:
            - output_dim: number of hidden neurons (output dimension), includes the linear nodes
            - use_linear: use linear nodes
            - use_sigmoid: use sigmoid nodes
            - use_gaussian: use gaussian nodes
            - zero_mean: make input zero mean
            - unit_var: make input unit variance

        This node needs training to rescale the input and to initialize the gaussian kernel
        """
        if not use_linear and not use_sigmoid and not use_gaussian:
            raise Exception('Use at least one type of kernel!')

        super(ELMNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.use_linear = use_linear
        self.use_sigmoid = use_sigmoid
        self.use_gaussian = use_gaussian
        self._max_mem = 20000 #maximum number of training samples in memory
        self.zero_mean = zero_mean
        self.unit_var = unit_var

        self._mean, self._std, self._len = 0, 0, 0
        self._x = None

    def _train(self, x):
        #iteratively calculate mean and std
        self._mean += np.sum(x, axis=0)
        self._std += np.sum(x**2, axis=0)
        self._len += len(x)
        #save data points to create gaussian kernel, is executed even if use_gauss=False to work well in combination with the optimizer
        if self._x == None:
            self._x = x
            self._n_batches = 1
        elif len(self._x) + len(x) > self._max_mem and self._n_batches > 2:
            # reduce memory use, originates from Matlab toolbox
            self._n_batches += 1
            self._x = self._x[np.random.permutation(len(self._x))[0:self._max_mem],:]
            n = np.round(0.0 + self._max_mem / self._n_batches)
            self._x[-n:,:] = x[np.random.permutation(len(x))[0:n],:]
        else:
            self._x = np.concatenate((self._x, x), axis=0)
            self._n_batches += 1

    def _sigm_initialize(self, N):
        N = int(N)
        self._n_sigm = N
        #magic 10**(-5) rescaling was used in orriginal matlab toolbox
        self._w_sigm = np.random.randn(self._input_dim, N) / np.sqrt(self.input_dim) #weights
        self._b_sigm = np.random.randn(1, N) #bias

    def _gauss_initialize(self, N):
        N = int(N)
        self._n_gauss = N
        self._x = (self._x - self._mean) / self._std
        Y = dist.pdist(self._x)
        a10 = sp.stats.scoreatpercentile(Y, 20) #copied from Matlab toolbox, don't know where the magic 20 comes from
        a90 = sp.stats.scoreatpercentile(Y, 60) #copied from Matlab toolbox, don't know where the magic 60 comes from
        MP = np.random.permutation(len(self._x))
        self._gauss_c = self._x[MP[0:N], :]
        self._gauss_sig2 = (np.random.rand(N,1) * (a90 - a10) + a10)**2

    def _stop_training(self):
        if self.zero_mean:
            self._mean /= self._len
        else:
            self._mean = 0
        if self.unit_var:
            self._std = np.sqrt(self._std / self._len - self._mean**2)
        else:
            self._std = 1
        self.initialize()
        if np.rank(self._x) < self.input_dim:
            import warnings
            warnings.warn('One or more inputs seem correlated. Better performance is often achieved with uncorrelated inputs.')

    def initialize(self):
        N = self._output_dim - (self._input_dim) * self.use_linear
        if self.use_sigmoid and self.use_gaussian:
            self._sigm_initialize(np.ceil(N / 2.0))
            self._gauss_initialize(np.floor(N / 2.0))
        elif self.use_sigmoid:
            self._sigm_initialize(N)
        elif self.use_gaussian:
            self._gauss_initialize(N)
        elif self._output_dim != self._input_dim:
            self._output_dim = self._input_dim
            import warnings
            warnings.warn('Changed output_dim because only linear nodes are used and output_dim != input_dim.')

    def _gauss_func(self, x, c, sig2):
        return np.exp(- np.mean((x - c)**2, axis=1) / sig2)

    def _execute(self, x):
        x = (x - self._mean) / self._std
        y = np.zeros((len(x), self._output_dim))
        s = 0
        if self.use_linear:
            y[:,0:self._input_dim] = x
            s = self._input_dim
        if self.use_sigmoid:
            y[:,s:s+self._n_sigm] = np.tanh(np.dot(x, self._w_sigm) + self._b_sigm)
            s += self._n_sigm
        if self.use_gaussian:
            for i in range(self._n_gauss):
                y[:,s+i] = self._gauss_func(x, self._gauss_c[i,:], self._gauss_sig2[i,:])
        return y
