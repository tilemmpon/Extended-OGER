import Oger
import mdp.utils
import numpy as np
import collections
import inspect
from calculate_pi_digits import get_pi_digits_in_array, load_n_digits_of_pi
try:
    import cudamat as cm
except:
    pass

# TODO: MDP parallelization assumes that nodes are state-less when executing! The current ReservoirNode
# does not adhere to this and therefor are not parallelizable. A solution is to make the state local. 
# But this again breaks if we start doing some form of on-line learning. In this case we need, or, 
# fork-join but this only work when training (and joining is often not possible), or do not have 
# fork-join in training mode which will force synchronous execution.

# TODO: leaky neuron is also broken when parallel! 

class ReservoirNode(mdp.Node):
    """
    A standard (ESN) reservoir node.
    """

    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
                 nonlin_func=None, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        super(ReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        # Set all object attributes
        # Scaling for input weight matrix
        self.input_scaling = input_scaling
        # Scaling for bias weight matrix
        self.bias_scaling = bias_scaling
        # Spectral radius scaling
        self.spectral_radius = spectral_radius
        # Instance ID, used for making different reservoir instantiations with the same parameters
        # Can be ranged over to simulate different 'runs'
        self._instance = _instance
        # Non-linear function
        if nonlin_func == None:
            self.nonlin_func = Oger.utils.TanhFunction
        else:
            self.nonlin_func = nonlin_func


        # Store any externally passed initialization values for w, w_in and w_bias
        self.w_in_initial = w_in
        self.w_initial = w
        self.w_bias_initial = w_bias

        # Fields for allocating reservoir weight matrix w, input weight matrix w_in
        # and bias weight matrix w_bias
        self.w_in = np.array([])
        self.w = np.array([])
        self.w_bias = np.array([])

        self.reset_states = reset_states

        self._is_initialized = False

        if input_dim is not None and output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()


    # Override the standard output_dim getter and setter property, 
    # to enable changing the output_dim (i.e. the number
    # of neurons) afterwards during optimization
    def get_output_dim(self): 
            return self._output_dim

    def set_output_dim(self, value): 
            self._output_dim = value
    output_dim = property(get_output_dim, set_output_dim, doc="Output dimensions")

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def initialize(self):
        """ Initialize the weight matrices of the reservoir node. If no
        arguments for w, w_in and w_bias matrices were given at construction
        time, they will be created as follows:
            - input matrix : input_scaling * weights randomly drawn from
              the set {-1, 1}
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius

        For more control over the weight matrix creation, you can also specify the
        w, w_in or w_bias as numpy arrays or as callables (i.e. functions). In the latter
        case, the functions for w, w_in, w_bias should accept the following arguments:
        - w = w_init_function(output_dim, spectral_radius)
        - w_in = w_in_init_function(output_dim, input_dim, input_scaling)
        - w_bias = w_bias_init_function(output_dim, bias_scaling)
        The weight matrices are created either at instantiation (if input_dim and output_dim are
        both given to the constructor), or during the first call to execute.
        """
        if self.input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if self.output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')

        # Initialize input weight matrix
        if self.w_in_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_in = self.input_scaling * (mdp.numx.random.randint(0, 2, (self.output_dim, self.input_dim)) * 2 - 1)
        else:
            if callable(self.w_in_initial):
                self.w_in = self.w_in_initial(self.output_dim, self.input_dim, self.input_scaling) # If it is a function, call it
            else:
                self.w_in = self.w_in_initial.copy() # else just copy it
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_in.shape != (self.output_dim, self.input_dim):
            exception_str = 'Shape of given w_in does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_in: ' + str(self.w_in.shape)
            raise mdp.NodeException(exception_str)

        # Initialize bias weight matrix
        if self.w_bias_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_bias = self.bias_scaling * (mdp.numx.random.rand(1,self.output_dim) * 2 - 1)
        else:
            if callable(self.w_bias_initial):
                self.w_bias = self.w_bias_initial(self.output_dim, self.bias_scaling) # If it is a function, call it
            else:
                self.w_bias = self.w_bias_initial.copy()   # else just copy it

        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_bias.shape != (1, self.output_dim):
            exception_str = 'Shape of given w_bias does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_bias: ' + str(self.w_bias.shape)
            raise mdp.NodeException(exception_str)

        # Initialize reservoir weight matrix
        if self.w_initial is None:
            self.w = mdp.numx.random.randn(self.output_dim, self.output_dim)
            # scale it to spectral radius
            self.w *= self.spectral_radius / Oger.utils.get_spectral_radius(self.w)
        else:
            if callable(self.w_initial):
                self.w = self.w_initial(self.output_dim, self.spectral_radius) # If it is a function, call it
            else:
                self.w = self.w_initial.copy()   # else just copy it

        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w.shape != (self.output_dim, self.output_dim):
            exception_str = 'Shape of given w does not match input/output dimensions of node. '
            exception_str += 'Output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w: ' + str(self.w.shape)
            raise mdp.NodeException(exception_str)

        self.initial_state = mdp.numx.zeros((1, self.output_dim))
        self.states = mdp.numx.zeros((1, self.output_dim))

        self._is_initialized = True


    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()

        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            self.initial_state = mdp.numx.zeros((1, self.output_dim))
        else:
            self.initial_state = mdp.numx.atleast_2d(self.states[-1, :])

        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim))))

        if inspect.isclass(self.nonlin_func):
            nonlinear_function_pointer = self.nonlin_func.f
        else:
            nonlinear_function_pointer = self.nonlin_func

        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            states[n + 1, :] = nonlinear_function_pointer(mdp.numx.dot(self.w, states[n, :]) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)
            self._post_update_hook(states, x, n)

        # Save the state for re-initialization in case reset_states = False
        self.states = states[1:, :]

        # Return the whole state matrix except the initial state
        return self.states

    def _post_update_hook(self, states, input, timestep):
        """ Hook which gets executed after the state update equation for every timestep. Do not use this to change the state of the 
            reservoir (e.g. to train internal weights) if you want to use parallellization - use the TrainableReservoirNode in that case.
        """
        pass


class LeakyReservoirNode(ReservoirNode):
    """Reservoir node with leaky integrator neurons (a first-order low-pass filter added to the output of a standard neuron). 
    """

    def __init__(self, leak_rate=1.0, *args, **kwargs):
        """Initializes and constructs a random reservoir with leaky-integrator neurons.
           Parameters are:
                - input_dim: input dimensionality
                - output_dim: output_dimensionality, i.e. reservoir size
                - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
                - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
                - input_scaling: scaling of the input weight matrix, default: 1
                - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
                - leak_rate: if 1 it is a standard neuron, lower values give slower dynamics

           Weight matrices are either generated randomly or passed at construction time.  If w, w_in or w_bias are not given in the constructor, they are created randomly:
                - input matrix : input_scaling * uniform weights in [-1, 1]
                - bias matrix :  bias_scaling * uniform weights in [-1, 1]
                - reservoir matrix: gaussian weights rescaled to the desired spectral radius

           If w, w_in or w_bias were given as a numpy array or a function, these will be used as initialization instead.
        """
        super(LeakyReservoirNode, self).__init__(*args, **kwargs)

        # Leak rate, if 1 it is a standard neuron, lower values give slower dynamics 
        self.leak_rate = leak_rate

    def _post_update_hook(self, states, input, timestep):
        states[timestep + 1, :] = (1 - self.leak_rate) * states[timestep, :] + self.leak_rate * states[timestep + 1, :]

class SparseReservoirNode(ReservoirNode):
    """ Initializes and constructs a random reservoir with sparse uniformly random weight matrices in [-1, 1].
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
            - fan_in_w: average number of non-zero incoming weights in the reservoir weight matrix. Default value 10.
            - fan_in_i: average number of non-zero incoming weights in the input weight matrix. Default value 1.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        This Node type is not parallelizable using threads.
        """
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
             nonlin_func=np.tanh, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
             w_bias=None, fan_in_w = 10, fan_in_i = 1):

        self.fan_in_w = fan_in_w
        self.fan_in_i = fan_in_i

        # Call the parent init function to set the weight matrix generation functions
        super(SparseReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, spectral_radius=spectral_radius,
                 nonlin_func=nonlin_func, reset_states=reset_states, bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, _instance=_instance,
                 w_in=self.sparse_w_in, w_bias=w_bias, w=self.sparse_w)

    # Use sparse matrices for generating the weights
    def sparse_w(self, out_size, specrad):
        import scipy.sparse
        from scipy.sparse.linalg import eigs
        converged = False
        # Initialize reservoir weight matrix
        nrentries = mdp.numx.int32(out_size * self.fan_in_w)
        # Keep generating random matrices until convergence
        while not converged:
            try:
                #$%ij = mdp.numx.zeros((2,nrentries))
                ij = mdp.numx.random.randint(0,out_size,(2,nrentries))
                datavec =  mdp.numx.random.randn(nrentries)
                w = scipy.sparse.csc_matrix((datavec, ij),dtype=self._dtype, shape=(out_size, out_size))
                we = eigs(w,return_eigenvectors=False,k=3)
                converged = True
                w *= (specrad / mdp.numx.amax(mdp.numx.absolute(we)))
            except:
                pass
        return w

    def sparse_w_in(self, out_size, in_size, scaling):
        import scipy.sparse
        # Initialize reservoir weight matrix
        nrentries = mdp.numx.int32(out_size * self.fan_in_i)
        # Keep generating random matrices until convergence
        ij = mdp.numx.zeros((2,nrentries))
        ij[0,:] = mdp.numx.random.randint(0,out_size,(1,nrentries))
        ij[1,:] = mdp.numx.random.randint(0,in_size,(1,nrentries))
        datavec =  mdp.numx.random.rand(nrentries)
        w = scaling * scipy.sparse.csc_matrix((datavec, ij),dtype=self._dtype, shape=(out_size, in_size))
        return w

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()

        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            self.initial_state = mdp.numx.zeros((1, self.output_dim))
        else:
            self.initial_state = mdp.numx.atleast_2d(self.states[-1, :])

        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim))))

        nonlinear_function_pointer = self.nonlin_func

        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            states[n + 1, :] = nonlinear_function_pointer(self.w*states[n, :] + self.w_in* x[n, :] + self.w_bias)
            self._post_update_hook(states, x, n)

        # Save the state for re-initialization in case reset_states = False
        self.states = states[1:, :]

        # Return the whole state matrix except the initial state
        return self.states

class SparseLeakyReservoirNode(SparseReservoirNode, LeakyReservoirNode):
    """ Initializes and constructs a random reservoir with sparse uniformly random weight matrices in [-1, 1].
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
            - fan_in_w: average number of non-zero incoming weights in the reservoir weight matrix. Default value 10.
            - fan_in_i: average number of non-zero incoming weights in the input weight matrix. Default value 1.
            - leak_rate: if 1 it is a standard neuron, lower values give slower dynamics
        In case reset_states = False, note that because state needs to be stored in the Node object,
        This Node type is not parallelizable using threads.
        """
    def __init__(self, leak_rate = 1, input_dim=None, output_dim=None, spectral_radius=0.9,
             nonlin_func=np.tanh, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
             w_bias=None, fan_in_w = 10, fan_in_i = 1):
        # Leak rate, if 1 it is a standard neuron, lower values give slower dynamics 
        super(SparseLeakyReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, spectral_radius=spectral_radius, nonlin_func=nonlin_func, reset_states=reset_states, bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, _instance=_instance, w_bias=w_bias, fan_in_w = fan_in_w, fan_in_i = fan_in_i )
        self.leak_rate = leak_rate


class BandpassReservoirNode(ReservoirNode):
    """Reservoir node with bandpass neurons (an Nth-order band-pass filter added to the output of a standard neuron).
    """
    def __init__(self, b=mdp.numx.array([[1]]), a=mdp.numx.array([[0]]), *args, **kwargs):
        """Initializes and constructs a random reservoir with band-pass neurons.
           Parameters are:
                - input_dim: input dimensionality
                - output_dim: output_dimensionality, i.e. reservoir size
                - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
                - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
                - input_scaling: scaling of the input weight matrix, default: 1
                - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
                - b: array of coefficients for the numerator of the IIR filter
                - a: array of coefficients for the denominator of the IIR filter
           Weight matrices are either generated randomly or passed at construction time.
           if w, w_in or w_bias are not given in the constructor, they are created randomly:
               - input matrix : input_scaling * uniform weights in [-1, 1]
               - bias matrix :  bias_scaling * uniform weights in [-1, 1]
               - reservoir matrix: gaussian weights rescaled to the desired spectral radius
           If w, w_in or w_bias were given as a numpy array or a function, these
           will be used as initialization instead.
        """
        self.a = a
        self.b = b
        super(BandpassReservoirNode, self).__init__(*args, **kwargs)
        self.input_buffer = collections.deque([mdp.numx.zeros(self.output_dim)] * self.b.shape[1], maxlen=self.b.shape[1])
        self.output_buffer = collections.deque([mdp.numx.zeros(self.output_dim)] * self.a.shape[1], maxlen=self.a.shape[1])

    def _post_update_hook(self, states, input, timestep):
        self.input_buffer.appendleft(states[timestep + 1, :])
        t1 = mdp.numx.sum(self.b * mdp.numx.array(self.input_buffer).T, axis=1)
        t2 = mdp.numx.sum(self.a * mdp.numx.array(self.output_buffer).T, axis=1)
        states[timestep + 1, :] = t1 - t2
        self.output_buffer.appendleft(states[timestep, :])

class TrainableReservoirNode(ReservoirNode):
    """A reservoir node that allows on-line training of the internal connections. Use
    this node for this purpose instead of implementing the _post_update_hook in the
    normal ReservoirNode as this is incompatible with parallelization. 
    """
    def is_trainable(self):
        return True

    def _train(self, x):
        states = self._execute(x)
        self._post_train_hook(states, input)

    def _post_update_hook(self, states, input, timestep):
        super(TrainableReservoirNode, self)._post_update_hook(states, input, timestep)
        if self.is_training():
            self._post_train_update_hook(states, input, timestep)

    def _post_train_update_hook(self, states, input, timestep):
        """Implement this function for on-line training after each time-step
        """
        pass

    def _post_train_hook(self, states, input):
        """Implement this function for training after each time-series
        """
        pass

class HebbReservoirNode(TrainableReservoirNode):
    """This node does nothing good, it is just a demo of training a reservoir.
    """
    def _post_train_update_hook(self, states, input, timestep):
        self.w -= 0.01 * mdp.utils.mult(states[timestep + 1:timestep + 2, :].T, states[timestep:timestep + 1, :])
        self.w_in -= 0.01 * mdp.utils.mult(states[timestep + 1:timestep + 2, :].T, input[timestep:timestep + 1, :])
        self.w_bias -= 0.01 * states[timestep + 1, :];


class GaussianIPReservoirNode(TrainableReservoirNode):
    ''' This ReservoirNode is adaptable using IP. Only works for tanh nonlinearities.
        This node is trainable. Can be used for pre-adaptation.
        See Verstraeten, D., 'Reservoir Computing: computation with dynamical Systems', PhD Thesis, Ghent University for theory.
        
        Constructor parameters (in addition to the standard reservoir ones):
            - mu: desired mean of the output distribution (default: 0)
            - sigma_squared: desired variance of the output distribution (default: .04)
            - eta: learning rate (default: .0001)
    '''
    def __init__(self, eta=.0001, mu=0, sigma_squared=.04, keep_parameter_history=True, *args, **kwargs):
        super(GaussianIPReservoirNode, self).__init__(*args, **kwargs)
        self.a = np.ones(self.output_dim)
        self.keep_parameter_history = keep_parameter_history
        self.eta = eta
        self.mu = mu
        self.sigma_squared = sigma_squared

        if self.keep_parameter_history:
            self.aa = []
            self.daa = []
            self.dbb = []
            self.bb = []

    def _post_train_update_hook(self, states, input, timestep):
        ''' Compute the new a (gain) and b (bias) parameters for the reservoir, and apply to w and w_bias
        '''
        # Store the original w_res to be able to apply a
        if not hasattr(self, 'w_orig'):
            self.w_orig = self.w

        # Store some variables for easy access
        y = states[timestep + 1, :]
        x = np.arctanh(y)
        m = self.mu
        s = self.sigma_squared

        # Compute parameter deltas
        db = self.eta * (m / s - y / s * (2 * s + 1 - y ** 2 + m * y))
        da = self.eta / self.a + db * x

        # Apply parameter changes
        self.a += da
        self.w = self.w_orig * self.a
        self.w_bias += db

        # Should we store the history of the parameters?
        if self.keep_parameter_history:
            self.aa.append(self.a.copy())
            self.daa.append(da)
            self.dbb.append(db)
            self.bb.append(self.w_bias.copy())

def get_specrad(Ac):
        """Get spectral radius of A using the power method."""

        m_size = Ac.shape[0]

        x = np.random.normal(0, 1, (m_size, 1))

        x = x / np.linalg.norm(x)
        x = cm.CUDAMatrix(x)

        y = cm.empty((m_size, 1))
        diff = 200
        eps = 1e-3
        b = 1e10
        c = 1e9
        max_its = 1e6

        n_its = 0

        while diff > eps and n_its < max_its:
            cm.dot(Ac, x, target=y)
            norm = y.euclid_norm()
            y.divide(norm, target=x)
            a = cm.dot(y.T, x).asarray()
            c = cm.dot(x.T, x).asarray()
            diff = np.abs(a - b)
            b = float(a)
            n_its += 1

        specrad = float(a / c)
        print 'Spectral radius:', specrad, 'Number of iterations:', n_its
        return float(a / c)


class CUDAReservoirNode(mdp.Node):
    def __init__(self, input_dim, output_dim, spectral_radius=.9, leak_rate=1,
                                             input_scaling=1, bias_scaling=0):
        super(CUDAReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim,)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leak_rate = leak_rate
        w = mdp.numx.random.normal(0, 1, (output_dim, output_dim))
        w_in = mdp.numx.random.uniform(-1, 1, (output_dim, input_dim)) * input_scaling
        if output_dim < 1500:
            l = mdp.numx.linalg.eigvals(w)
            r = mdp.numx.amax(mdp.numx.absolute(l))
            w = w * (spectral_radius / r)
            self.w = cm.CUDAMatrix(w)
        else:
            self.w = cm.CUDAMatrix(w)
            r = get_specrad(self.w)
            self.w.mult(spectral_radius / r)
        bias = mdp.numx.random.normal(0, 1, (output_dim, 1)) * bias_scaling
        self.w_in = cm.CUDAMatrix(w_in)
        self.bias = cm.CUDAMatrix(bias)
        self.current_state = cm.empty((self.output_dim, 1))
        self.new_state = cm.empty((self.output_dim, 1))


    def _execute(self, x):
        n = x.shape[0]
        # Do everything in transpose because row indexing is very expensive.
        x_T = x.transpose()
        self.states = cm.empty((self.output_dim, n + 1))

        self.states.set_col_slice(0, 1, cm.CUDAMatrix(mdp.numx.zeros((self.output_dim, 1))))

        for i in range(n):

            self.current_state = self.states.get_col_slice(i, i + 1)
            self.new_state = self.states.get_col_slice(i + 1, i + 2)

            cm.dot(self.w, self.current_state, self.new_state)
            self.new_state.add_dot(self.w_in, x_T.get_col_slice(i, i + 1))

            self.new_state.apply_tanh()

            # Note that the states are shifted in time and that the first state
            # is zero.
            states_out = self.states.get_col_slice(0, n)

        return states_out.transpose()


class DelayLineReservoirNode(ReservoirNode):
    """
    A standard delay line reservoir (DLR) node as described in "Rodan, A., & 
    Tino, P. (2011). Minimum complexity echo state network. IEEE transactions 
    on neural networks, 22(1), 131-144.".
    """
    
    def __init__(self, input_dim=None, 
        output_dim=None, nonlin_func=np.tanh, reset_states=True, 
        bias_scaling=0, connections_scaling=1, input_scaling=1, 
        dtype='float64', _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a minimum complexity reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - connections_scaling: scaling of the weight matrix, default: 1
            - input_scaling: scaling of the input weight matrix, default: 1
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated deterministically or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created deterministically:
            - input matrix : input_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - bias matrix :  bias_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - reservoir matrix: connections_scaling * weights of lower subdiagonal (of  value 1) and all other zero.
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
            
        # create the input weight matrix
        if w_in is None:
            w_in = input_scaling * (mdp.numx.random.binomial(1, 0.5, 
                                            [output_dim, input_dim]) * 2 - 1)
            
        # create the reservoir weight matrix
        if w is None:
            w = mdp.numx.zeros([output_dim, output_dim])
            for i in range(output_dim - 1):
                w[i + 1, i] = connections_scaling
        
        # create the bias weight matrix
        if w_bias is None:
            w_bias = bias_scaling * (mdp.numx.random.binomial(
                        1, 0.5, output_dim)[mdp.numx.newaxis, :] * 2 - 1)
        
        # invoke default reservoir constructor using created weight matrices
        super(DelayLineReservoirNode, self).__init__(input_dim=input_dim, 
        output_dim=output_dim, spectral_radius=0.9, 
        nonlin_func=nonlin_func, reset_states=reset_states, 
        bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, 
        _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)
        

class DelayLineWithFeedbackReservoirNode(DelayLineReservoirNode):
    """
    A standard delay line reservoir with feedback connections (DLRB) node as 
    described in "Rodan, A., & Tino, P. (2011). Minimum complexity echo state 
    network. IEEE transactions on neural networks, 22(1), 131-144.".
    """
    
    def __init__(self, input_dim=None, 
        output_dim=None, nonlin_func=np.tanh, reset_states=True, 
        bias_scaling=0, feedforward_connections_scaling=1, 
        feedback_connections_scaling=1, input_scaling=1, dtype='float64', 
        _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a minimum complexity reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - feedforward_connections_scaling: scaling of the feedforward connections of the weight matrix, default: 1
            - feedback_connections_scaling: scaling of the feedback connections of the weight matrix, default: 1
            - input_scaling: scaling of the input weight matrix, default: 1
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated deterministically or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created deterministically:
            - input matrix : input_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - bias matrix :  bias_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - reservoir matrix: feedforward_connections_scaling * weights of lower subdiagonal (of value 1), feedback_connections_scaling * upper subdiagonal (of value 1) and all other zero.
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
            
        # create the weight matrix                    
        if w is None:
            w = mdp.numx.zeros([output_dim, output_dim])
            for i in range(output_dim - 1):
                w[i + 1, i] = feedforward_connections_scaling
                w[i, i + 1] = feedback_connections_scaling
                
        # invoke default DLR constructor using created weight matrix
        super(DelayLineWithFeedbackReservoirNode, self).__init__(
        input_dim=input_dim, output_dim=output_dim, nonlin_func=nonlin_func, 
        reset_states=reset_states, bias_scaling=bias_scaling, 
        input_scaling=input_scaling, dtype=dtype, _instance=_instance, 
        w_in=w_in, w=w, w_bias=w_bias)


class SimpleCycleReservoirNode(DelayLineReservoirNode):
    """
    A standard simple cycle reservoir (SCR) node as described in "Rodan, A., & 
    Tino, P. (2011). Minimum complexity echo state network. IEEE transactions 
    on neural networks, 22(1), 131-144.".
    """
    
    def __init__(self, input_dim=None, 
        output_dim=None, nonlin_func=np.tanh, reset_states=True, 
        bias_scaling=0, connections_scaling=1, input_scaling=1, 
        determistic_input_matrix=False, pi_file_loc='pi_10_mil_digits.txt', 
        dtype='float64', _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a minimum complexity reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - determistic_input_matrix: should the input and bias weight matrix be constructed randomly or deterministically? Default: False
            - pi_file_loc: location and name of the pi file, default: 'pi_10_mil_digits.txt' (point it to correct location)
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - connections_scaling: scaling of the weight matrix, default: 1
            - input_scaling: scaling of the input weight matrix, default: 1
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated deterministically or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created deterministically:
            - input matrix : if deterministic input_scaling * weights of absolute value 1. Sign of weights according to pi digits.
                            if random input_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - bias matrix :  if deterministic bias_scaling * weights of absolute value 1. Sign of weights according to pi digits.
                            if random bias_scaling * weights of absolute value 1. Sign of weights determined randomly by a random draw from Bernoulli distribution of mean 1/2 (unbiased coin).
            - reservoir matrix: connections_scaling * weights of lower subdiagonal and upper right weight value (of value 1) and all other zero.
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
        
        # create the input weight matrix if deterministic
        if determistic_input_matrix and w_in is None:
            w_in = input_scaling * mdp.numx.where(
                get_pi_digits_in_array([output_dim, input_dim]) < 5, -1, 1)
            
        # create the reservoir weight matrix
        if w is None:
            w = mdp.numx.zeros([output_dim, output_dim])
            for i in range(output_dim - 1):
                w[i + 1, i] = connections_scaling
                
            w[0, output_dim - 1] = connections_scaling
            
        # create the bias weight matrix if deterministic
        if determistic_input_matrix and w_bias is None:
            w_bias = bias_scaling * mdp.numx.where(mdp.numx.array(
            load_n_digits_of_pi(output_dim, pi_file_loc)) < 5, 1, -1)[mdp.numx.newaxis, :]

        # invoke default DLR constructor using created weight matrix
        super(SimpleCycleReservoirNode, self).__init__(
        input_dim=input_dim, output_dim=output_dim, nonlin_func=nonlin_func, 
        reset_states=reset_states, bias_scaling=bias_scaling, 
        input_scaling=input_scaling, dtype=dtype, _instance=_instance, 
        w_in=w_in, w=w, w_bias=w_bias)


class CycleReservoirWithJumpsNode(SimpleCycleReservoirNode):
    """
    A standard Cycle Reservoir with Jumps (CRJ) node as described in "Rodan, A.
    , & Tiňo, P. (2012). Simple deterministically constructed cycle reservoirs 
    with regular jumps. Neural computation, 24(7), 1822-1852.".
    """
    
    def __init__(self, input_dim=None, output_dim=None, jump_size = 3, 
                 nonlin_func=np.tanh, reset_states=True, bias_scaling=0, 
                 cycle_connections_scaling=1, jumps_connections_scaling=1, 
                 input_scaling=1, pi_file_loc='pi_10_mil_digits.txt', 
                 dtype='float64', _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a deterministic reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - jump_size: size of the shortcuts (jumps), default: 3
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - cycle_connections_scaling: scaling of the uni-directional cycle weights of the reservoir, default: 1
            - jumps_connections_scaling: bi-directional shortcuts (jumps) weights of the reservoir, default: 1
            - input_scaling: scaling of the input weight matrix, default: 1
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
            - pi_file_loc: location and name of the pi file, default: 'pi_10_mil_digits.txt' (point it to correct location)
        
        Weight matrices are either generated deterministically or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created deterministically:
            - input matrix : input_scaling * weights of absolute value 1. Sign of weights according to pi digits.
            - bias matrix :  bias_scaling * weights of absolute value 1. Sign of weights according to pi digits.
            - reservoir matrix: connections_scaling * weights of lower subdiagonal (of  absolute value 1) and all other zero. Sign of lower subdiagonal weights according to pi digits. 
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
            
        if jump_size >= output_dim / 2 or jump_size < 1:
            raise mdp.NodeException('Jump size must be between [1, output_dim / 2).')
                    
        # create the reservoir weight matrix
        if w is None:
            w = mdp.numx.zeros([output_dim, output_dim])            
            # create the uni-directional cycle
            for i in range(output_dim - 1):
                w[i + 1, i] = cycle_connections_scaling
            w[0, output_dim - 1] = cycle_connections_scaling
            # create the bi-directional jumps
            for i in range(0, output_dim - jump_size, jump_size):
                w[i, i + jump_size] = jumps_connections_scaling
                w[i + jump_size, i] = jumps_connections_scaling
                
        # invoke default SCR constructor using created weight matrix
        super(CycleReservoirWithJumpsNode, self).__init__(
        input_dim=input_dim, output_dim=output_dim, nonlin_func=nonlin_func, 
        reset_states=reset_states, bias_scaling=bias_scaling, 
        determistic_input_matrix=True, input_scaling=input_scaling, 
        pi_file_loc=pi_file_loc, dtype=dtype, _instance=_instance, w_in=w_in, 
        w=w, w_bias=w_bias)
        
        
        
class FeedForwardESNReservoir(ReservoirNode):
    """
    A standard Feed-Forward ESN reservoir model as described in "Čerňanský, M., 
    & Tiňo, P. (2008, September). Predictive modeling with echo state networks. 
    In International Conference on Artificial Neural Networks (pp. 778-787). 
    Springer Berlin Heidelberg.".
    """
    
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
                 nonlin_func=np.tanh, reset_states=True, bias_scaling=0, 
                 input_scaling=1, dtype='float64', _instance=0, w_in=None, 
                 w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius but keeping only elements below diagonal
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
          
        # invoke default reservoir constructor to create weight matrices
        super(FeedForwardESNReservoir, self).__init__(input_dim=input_dim, 
        output_dim=output_dim, spectral_radius=spectral_radius, 
        nonlin_func=nonlin_func, reset_states=reset_states, 
        bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, 
        _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)
        
        # keep the elements below diagonal for reservoir weight matrix
        self.w = mdp.numx.tril(self.w, -1)
        
        
class SparseAndOrthogonalMatricesReservoir(ReservoirNode):
    """
    A standard Sparse and Orthogonal Reservoir Matrices (SORM) model  as 
    described in "Strauss, T., Wustlich, W., & Labahn, R. (2012). Design 
    strategies for weight matrices of echo state networks. Neural computation, 
    24(12), 3246-3276.".
    """

    def __init__(self, input_dim=None, output_dim=None, reservoir_density=0.8,
                 spectral_radius=0.9, nonlin_func=np.tanh, reset_states=True, 
                 bias_scaling=0, input_scaling=1, dtype='float64', 
                 _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - reservoir_density: desired density of the reservoir weight matrix, default value: 0.8
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: multiplication of an initial permutation matrix with sparse orthogonal matrices Q
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
        
        if w is None:
            w = mdp.numx.eye(output_dim)            
            mdp.numx.random.shuffle(w)
            while (not self._is_permuation_matrix(w)):  
                mdp.numx.random.shuffle(w)
            
            density = 0
            while density < reservoir_density:
                w = self._create_Q_matrix(output_dim).dot(w)
            
                density = len(mdp.numx.nonzero(w)[0]) / float(w.shape[0] * w.shape[1])
                
            # scale weight matrix to desired spectral radius (step 3 of Jaeger algorithm)
            w *= spectral_radius
        
        # invoke default reservoir constructor using created weight matrix
        super(SparseAndOrthogonalMatricesReservoir, self).__init__(
        input_dim=input_dim, output_dim=output_dim, spectral_radius=spectral_radius, 
        nonlin_func=nonlin_func, reset_states=reset_states, 
        bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, 
        _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)
    
    
    def _is_permuation_matrix(self, x):
        """
        Checks if a matrix x is a permutation matrix.
        
        Parameters:
        -----------
        x : numpy array
        """
        # source: http://stackoverflow.com/questions/28895894/how-to-quickly-determine-if-a-matrix-is-a-permutation-matrix
        x = mdp.numx.asanyarray(x)
        return (x.ndim == 2 and x.shape[0] == x.shape[1] and
                (x.sum(axis=0) == 1).all() and 
                (x.sum(axis=1) == 1).all() and
                ((x == 1) | (x == 0)).all())
                
            
    def _create_Q_matrix(self, size):
        """
        Creates a sparse orthogonal matrix Q.
        
        Parameters:
        -----------
        size : int
            size of the matrix
        """
        Q = mdp.numx.eye(size)
        h = mdp.numx.random.randint(0, size)
        k = mdp.numx.random.randint(0, size)
        while (h == k):
            h = mdp.numx.random.randint(0, size)
        if k > h:
            k, h = h, k
        phi = mdp.numx.random.random() * (2 * mdp.numx.pi)
        Q[h, h] = mdp.numx.cos(phi)
        Q[h, k] = -mdp.numx.sin(phi)
        Q[k, h] = mdp.numx.sin(phi)
        Q[k, k] = mdp.numx.cos(phi)
        return Q
        
class CyclicSORMsReservoir(SparseAndOrthogonalMatricesReservoir):
    """
    A standard Cyclic Sparse and Orthogonal Reservoir Matrices (CycleSORMs) 
    model  as described in "Strauss, T., Wustlich, W., & Labahn, R. (2012). 
    Design strategies for weight matrices of echo state networks. Neural 
    computation, 24(12), 3246-3276.".
    """
    def __init__(self, input_dim=None, output_dim=None, v_mat_density=0.3,
                 spectral_radius=0.9, nonlin_func=np.tanh, reset_states=True, 
                 bias_scaling=0, input_scaling=1, dtype='float64', 
                 _instance=0, w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - v_mat_density: desired density of the sparse and orthogonal transformation matrix V, default value: 0.3
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: multiplication of an initial permutation matrix with sparse and orthogonal transformation matrix V, w = V*P*V.T
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        if input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
        
        if w is None:
            w = mdp.numx.eye(output_dim)
            mdp.numx.random.shuffle(w)
            while (not self._is_permuation_matrix(w)):  
                mdp.numx.random.shuffle(w)
            
            density = 0
            V = mdp.numx.eye(output_dim)
            while density < v_mat_density:
                V = V.dot(self._create_Q_matrix(output_dim))
            
                density = len(mdp.numx.nonzero(V)[0]) / float(V.shape[0] * V.shape[1])
                
            w = V.dot(w.dot(V.T))
                
            # scale weight matrix to desired spectral radius (step 3 of Jaeger algorithm)
            w *= spectral_radius
        
        # invoke default reservoir constructor using created weight matrix
        super(CysclicSORMsReservoir, self).__init__(
        input_dim=input_dim, output_dim=output_dim, spectral_radius=spectral_radius, 
        nonlin_func=nonlin_func, reset_states=reset_states, 
        bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, 
        _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)

class ExtendedReservoirNode(ReservoirNode):

    """
    A standard (ESN) reservoir node, that also allows direct mapping of input
    to output.
    """
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
                 nonlin_func=np.tanh, reset_states=True, bias_scaling=0,
                 input_scaling=1, input_to_output_map=False, 
                 input_to_output_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - input_to_output_map: should there be direct input to out mapping? Default: False
            - input_to_output_scaling: scaling of the mapping of input to output, default: 1
            - reset_states: should the reservoir states be reset to zero at each call of execute? Default True, set to False for use in FeedbackFlow
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        In case reset_states = False, note that because state needs to be stored in the Node object,
        this Node type is not parallelizable using threads.
        """
        self.input_to_output_map = input_to_output_map
        self.input_to_output_scaling = input_to_output_scaling
        
        super(ExtendedReservoirNode, self).__init__(input_dim=input_dim, 
        output_dim=output_dim, spectral_radius=spectral_radius, 
        nonlin_func=nonlin_func, reset_states=reset_states, 
        bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, 
        _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)
        

    def initialize(self):
        """ Initialize the weight matrices of the reservoir node. If no
        arguments for w, w_in and w_bias matrices were given at construction
        time, they will be created as follows:
            - input matrix : input_scaling * weights randomly drawn from
              the set {-1, 1}
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius

        For more control over the weight matrix creation, you can also specify the
        w, w_in or w_bias as numpy arrays or as callables (i.e. functions). In the latter
        case, the functions for w, w_in, w_bias should accept the following arguments:
        - w = w_init_function(output_dim, spectral_radius)
        - w_in = w_in_init_function(output_dim, input_dim, input_scaling)
        - w_bias = w_bias_init_function(output_dim, bias_scaling)
        The weight matrices are created either at instantiation (if input_dim and output_dim are
        both given to the constructor), or during the first call to execute.
        """
        if self.input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        if self.output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')

        # Initialize input weight matrix
        if self.w_in_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_in = self.input_scaling * (mdp.numx.random.randint(0, 2, (self.output_dim, self.input_dim)) * 2 - 1)
        else:
            if callable(self.w_in_initial):
                self.w_in = self.w_in_initial(self.output_dim, self.input_dim, self.input_scaling) # If it is a function, call it
            else:
                self.w_in = self.w_in_initial.copy() # else just copy it
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_in.shape != (self.output_dim, self.input_dim):
            exception_str = 'Shape of given w_in does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_in: ' + str(self.w_in.shape)
            raise mdp.NodeException(exception_str)

        # Initialize bias weight matrix
        if self.w_bias_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_bias = self.bias_scaling * (mdp.numx.random.rand(1,self.output_dim) * 2 - 1)
        else:
            if callable(self.w_bias_initial):
                self.w_bias = self.w_bias_initial(self.output_dim, self.bias_scaling) # If it is a function, call it
            else:
                self.w_bias = self.w_bias_initial.copy()   # else just copy it

        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_bias.shape != (1, self.output_dim):
            exception_str = 'Shape of given w_bias does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_bias: ' + str(self.w_bias.shape)
            raise mdp.NodeException(exception_str)

        # Initialize reservoir weight matrix
        if self.w_initial is None:
            self.w = mdp.numx.random.randn(self.output_dim, self.output_dim)
            # scale it to spectral radius
            self.w *= self.spectral_radius / Oger.utils.get_spectral_radius(self.w)
        else:
            if callable(self.w_initial):
                self.w = self.w_initial(self.output_dim, self.spectral_radius) # If it is a function, call it
            else:
                self.w = self.w_initial.copy()   # else just copy it

        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w.shape != (self.output_dim, self.output_dim):
            exception_str = 'Shape of given w does not match input/output dimensions of node. '
            exception_str += 'Output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w: ' + str(self.w.shape)
            raise mdp.NodeException(exception_str)

        if self.input_to_output_map:
            self.initial_state = mdp.numx.zeros((1, self.output_dim + self.input_dim))
            self.states = mdp.numx.zeros((1, self.output_dim + self.input_dim))
        else:
            self.initial_state = mdp.numx.zeros((1, self.output_dim))
            self.states = mdp.numx.zeros((1, self.output_dim))

        self._is_initialized = True
        
    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()

        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            if self.input_to_output_map:
                self.initial_state = mdp.numx.zeros((1, self.output_dim + self.input_dim))
            else:
                self.initial_state = mdp.numx.zeros((1, self.output_dim))
        else:
            self.initial_state = mdp.numx.atleast_2d(self.states[-1, :])

        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        if self.input_to_output_map:
            states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim + self.input_dim))))
        else:
            states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim))))

        if inspect.isclass(self.nonlin_func):
            nonlinear_function_pointer = self.nonlin_func.f
        else:
            nonlinear_function_pointer = self.nonlin_func

        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            if self.input_to_output_map:
                states[n + 1, :self.output_dim] = nonlinear_function_pointer(mdp.numx.dot(self.w, states[n, :self.output_dim]) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)
                states[n + 1, self.output_dim:] = self.input_to_output_scaling * x[n, :]
            else:
                states[n + 1, :] = nonlinear_function_pointer(mdp.numx.dot(self.w, states[n, :]) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)
            self._post_update_hook(states, x, n)

        # Save the state for re-initialization in case reset_states = False
        self.states = states[1:, :]

        # Return the whole state matrix except the initial state
        return self.states
