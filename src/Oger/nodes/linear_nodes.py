import mdp
import Oger
import numpy as np
import numpy.linalg as la
import os
import cPickle as pickle
import copy

class RidgeRegressionNode(mdp.Node):
    '''
    Ridge Regression Node that also optimizes the regularization parameter
    '''
    def __init__(self, ridge_param=mdp.numx.power(10, mdp.numx.arange(-15,15,0.2)), eq_noise_var=0, other_error_measure=None, cross_validate_function=None, low_memory=False, verbose=False, plot_errors=False, with_bias=True, clear_memory=True, input_dim=None, output_dim=None, dtype=None, *args, **kwargs):
        '''

        ridge_params contains the list of regularization parameters to be tested. If it is set to 0 no regularization
        or the eq_noise_var is used. Default 10^[-15:5:0.2].

        It is also possible to define an equivalent noise variance: the ridge parameter is set such that a
        regularization equal to a given added noise variance is achieved. Note that setting the ridge_param has
        precedence to the eq_noise_var and that optimizing the eq_noise_var is not yet supported.

        If an other_error_measure is used processing is slower! For classification for example one can use:
        other_error_measure = Oger.utils.threshold_before_error(Oger.utils.loss_01). Default None.

        cross_validation_function can be any function that returns a list of containing the cross validation sequence.
        The arguments for this function can be set using args and kwargs. n_samples is automatically set to the number
        of training examples given. Default Oger.evaluation.leave_one_out.

        low_memory=True Limits memory use to twice the size of the covariance matrix. It saves data in files instead of
        keeping them in memory, this is possible for up to 128 training examples. Default False

        verbose=True gives additional information about the optimization progress

        plot_errors=True gives a plot of the validation errors in function of log10(ridge_param). Default False

        with_bias=True adds an additional bias term. Default True.
        '''
        super(RidgeRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.ridge_param = ridge_param
        self.eq_noise_var = eq_noise_var
        self.other_error_measure = other_error_measure
        self.low_memory = low_memory
        self.verbose = verbose
        self.plot_errors = plot_errors
        self.with_bias = with_bias
        if cross_validate_function == None:
            cross_validate_function = Oger.evaluation.leave_one_out
        self.cross_validate_function = cross_validate_function
        self._args, self._kwargs = args, kwargs
        self.clear_memory = clear_memory

        self._xTx_list, self._xTy_list, self._yTy_list, self._len_list = [], [], [], []
        if other_error_measure:
            self._x_list = []
            self._y_list = []

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _train(self, x, y):
        y = y.astype(self.dtype) #avoid True + True != 2
        if self.with_bias:
            x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=self.dtype)), axis=1)
        if not self._output_dim == y.shape[1]:
            self._output_dim = y.shape[1]
        #Calculate the covariance matrices
        self._set('xTx', np.dot(x.T, x))
        self._set('xTy', np.dot(x.T, y))
        self._yTy_list.append(np.sum(y**2, axis=0))
        self._len_list.append(len(y))
        if self.other_error_measure:
            self._set('x', x)
            self._set('y', y)

    def _stop_training(self):
        if (type(self.ridge_param) is list or type(self.ridge_param) is np.ndarray) and len(self._xTx_list)>1:
            self._ridge_params = self.ridge_param
            if self.other_error_measure:
                calc_error = self._calc_other_error
            else:
                calc_error = self._calc_mse

            import time
            t_start = time.time()
            train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
            errors = np.zeros((len(self._ridge_params), self._output_dim))
            val_sets = range(len(train_samples))
            if self.verbose:
                val_sets = mdp.utils.progressinfo(val_sets, style='timer')
            for k in val_sets:
                errors += calc_error(train_samples[k], val_samples[k])
            errors /= len(train_samples)

            self.val_error, self.ridge_param = [], []
            for o in range(self._output_dim):
                r = mdp.numx.where(errors == np.nanmin(errors[:,o]))[0][-1]
                self.val_error.append(errors[r,o])
                self.ridge_param.append(self._ridge_params[r])
                if r==0 or r==len(self._ridge_params)-1:
                    import warnings
                    warnings.warn('The ridge parameter selected for output ' + str(o) + ' is ' + str(self.ridge_param[-1]) + '. This is the largest or smallest possible value from the list provided. Use larger or smaller ridge parameters to avoid this warning!')

            if self.verbose:
                print 'Total time:', time.time()-t_start, 's'
                print 'Found a ridge_param(s) =', self.ridge_param, 'with a validation error(s) of:', self.val_error
            if self.plot_errors:
                import pylab
                pylab.plot(np.log10(self._ridge_params),errors)
                pylab.show()
        else:
            if len(self._xTx_list)==1 and (type(self.ridge_param) is list or type(self.ridge_param) is np.ndarray) and len(self.ridge_param)>1:
                import warnings
                warnings.warn('Only one fold found, optimization is not supported. Instead no regularization or eq_noise_var is used!')
                self.ridge_param = 0
            elif self.ridge_param == 0:
                self.ridge_param = self.eq_noise_var**2 * np.sum(self._len_list)
            self.ridge_param = self.ridge_param * np.ones((self._output_dim,))

        self._final_training()
        self._clear_memory()

    def _execute(self, x):
        return np.dot(x, self.w).reshape((-1,self._output_dim)) + self.b

    def _get(self,name,l=None):
        l = list(l)
        ret = copy.copy(self._get_one(name, l.pop()))
        for i in l:
            if name.count('T') or name.count('len'):
                ret += self._get_one(name, i)
            else:
                ret = np.concatenate((ret, self._get_one(name, i)), axis=0)
        return ret

    def _get_one(self, name, i):
        t = getattr(self, '_' + name + '_list')
        if self.low_memory and not name.count('yTy') and not name.count('len'):
            t[i].seek(0)
            return pickle.load(t[i])
        else:
            return t[i]

    def _set(self,name,t,i=None):
        if self.low_memory and not name.count('yTy') and not name.count('len'):
            f = os.tmpfile()
            pickle.dump(t, f, protocol=-1)
            t = f
        if not i==None:
            getattr(self, '_' + name + '_list')[i] = t
        else:
            getattr(self, '_' + name + '_list').append(t)

    def _calc_mse(self, train, val, s=None):
        # Calculate the MSE for this validation set
        if s==None:
            s=range(self._input_dim + self.with_bias)
        errors = np.zeros((len(self._ridge_params), self._output_dim))
        D_t, C_t = la.eigh(self._get('xTx', train)[s,:][:,s] +  np.eye(len(s))) #reduce condition number
        D_t -= 1
        D_t[np.where(D_t<0)] = 0 #eigenvalues can only be positive
        D_t = 1 / (np.atleast_2d(D_t).T + np.atleast_2d(self._ridge_params))
        xTy_Ct = np.dot(C_t.T, self._get('xTy', train)[s,:])
        xTx_Cv = np.dot(C_t.T, np.dot(self._get('xTx', val)[s,:][:,s], C_t))
        xTy_Cv = 2 * np.dot(C_t.T, self._get('xTy', val)[s,:])
        # calculate error for all ridge params at once
        for o in range(self._output_dim):
            W_Ct = xTy_Ct[:,o:o+1] * D_t
            errors[:,o] += np.sum(W_Ct * (np.dot(xTx_Cv, W_Ct) - xTy_Cv[:,o:o+1]), axis=0)
        return (errors + self._get('yTy', val)) / self._get('len', val)

    def _calc_other_error(self, train, val, s=None):
        # Calculate error for this validation set using other error measure
        if s==None:
            s=range(self._input_dim + self.with_bias)
        errors = np.zeros((len(self._ridge_params), self._output_dim))
        xTx_t = self._get('xTx', train)[s,:][:,s]
        xTy_t = self._get('xTy', train)[s,:]
        x = self._get('x', val)[:,s]
        y = self._get('y', val)
        for r in range(len(self._ridge_params)):
            output = np.dot(x, la.solve(xTx_t + self._ridge_params[r] * np.eye(len(xTx_t)), xTy_t))
            for o in range(self._output_dim):
                errors[r,o] = self.other_error_measure(output[:,o], y[:,o])
        return errors

    def _final_training(self):
        # Calculate final weights
        xTx = self._get('xTx', range(len(self._xTx_list)))
        xTy = self._get('xTy', range(len(self._xTx_list)))
        W = np.zeros(xTy.shape)
        for o in range(self._output_dim):
            W[:,o] = la.solve(xTx + self.ridge_param[o] * np.eye(len(xTx)), xTy[:,o])
        if self.with_bias:
            self.b = W[-1, :]
            self.w = W[:-1, :]
        else:
            self.b = 0
            self.w = W

    def _clear_memory(self):
        if self.clear_memory:
            self._xTx_list, self._xTy_list, self._yTy_list, self._len_list = [],[],[],[]



class ClassReweightedRidgeRegressionNode(RidgeRegressionNode):
    '''
    Only 1 output is supported!!!
    '''
    def _get(self,name,l):
        if name.count('T'):
            n_pos = np.sum(np.array(self._n_list)[l], axis=0)[0]
            ret = super(ClassReweightedRidgeRegressionNode, self)._get(name, l) / n_pos
            n_neg = np.sum(np.array(self._n_list)[l], axis=0)[1]
            return ret + super(ClassReweightedRidgeRegressionNode, self)._get(name + 'n', l) / n_neg
        else:
            return super(ClassReweightedRidgeRegressionNode, self)._get(name, l)

    def _train(self,x,y):
        if len(y.shape)>1 and y.shape[1]>1:
            raise Exception('Only one output is supported!')

        # Calculate the covariance matrices
        if len(self._xTx_list) == 0:
            self._xTxn_list, self._xTyn_list, self._yTyn_list, self._n_list = [], [], [], []
            if not self._output_dim == y.shape[1]:
                self._output_dim = y.shape[1]
        if self.with_bias:
            x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=self.dtype)), axis=1)
        self._len_list.append(len(y))
        if self.other_error_measure:
            self._set('x', x)
            self._set('y', y)

        y = y>=0.000001
        xn = x[np.nonzero(1-y)[0], :]
        x = x[np.nonzero(y)[0], :]
        yn = (y[np.nonzero(1-y)[0], :]).astype(self.dtype) * 2 - 1 #avoid True + True != 2
        y = (y[np.nonzero(y)[0], :]).astype(self.dtype) * 2 - 1 #avoid True + True != 2

        self._set('xTx', np.dot(x.T, x))
        self._set('xTxn', np.dot(xn.T, xn))
        self._set('xTy', np.dot(x.T, y))
        self._set('xTyn', np.dot(xn.T, yn))
        self._yTy_list.append(np.sum(y**2, axis=0))
        self._yTyn_list.append(np.sum(yn**2, axis=0))
        self._n_list.append([float(len(y)), float(len(yn))])

    def _clear_memory(self):
        super(ClassReweightedRidgeRegressionNode, self)._clear_memory()
        self._xTxn_list = self._xTyn_list = self._yTyn_list = self._n_list = None


class BFSRidgeRegressionNode(RidgeRegressionNode):
    ''' Node that performs backward feature selection while optimizing the regularization parameter
    '''

    def __init__(self, ridge_param=mdp.numx.power(10, mdp.numx.arange(-15,15,0.5)), verbose=True, *args, **kwargs):
        '''
        see RidgeRegressionNode
        '''
        super(BFSRidgeRegressionNode, self).__init__(ridge_param=ridge_param, verbose=verbose, *args, **kwargs)


    def _stop_training(self):
        if not type(self.ridge_param) is list and not type(self.ridge_param) is np.ndarray:
            self._ridge_params = [self.ridge_param]
        else:
            self._ridge_params = self.ridge_param
        import time
        t_start = time.time()
        self._selected_inputs = range(self._input_dim + self.with_bias)
        train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
        if self.other_error_measure:
            calc_error = self._calc_other_error
        else:
            calc_error = self._calc_mse
        errors = np.zeros((len(self._ridge_params), self._output_dim))
        val_sets = range(len(train_samples))
        if self.verbose:
            val_sets = mdp.utils.progressinfo(val_sets, style='timer')
        for k in val_sets:
            errors += calc_error(train_samples[k], val_samples[k])
        errors = np.mean(errors, axis=1) / len(self._xTx_list)
        j = mdp.numx.where(errors == np.nanmin(errors))[0][-1]
        self.val_error = errors[j]
        self.ridge_param = self._ridge_params[j]
        if self.verbose:
            print 'Found a ridge_par =', self.ridge_param, 'with a val. error of:', self.val_error, 'Total nr. of selected inputs =', len(self._selected_inputs)-self.with_bias

        if self.other_error_measure:
            calc_error = self._calc_other_error_bfs
        else:
            calc_error = self._calc_mse_bfs
        while len(self._selected_inputs) > 1:
            errors = np.zeros((len(self._ridge_params), len(self._selected_inputs)-self.with_bias))
            val_sets = range(len(train_samples))
            if self.verbose:
                val_sets = mdp.utils.progressinfo(val_sets, style='timer')
            for k in val_sets:
                errors += calc_error(train_samples[k], val_samples[k])
            errors /= len(self._xTx_list)
            r, f = np.where(errors == np.nanmin(errors))
            r, f = r[-1], f[0]
            if errors[r,f] < self.val_error:
                self.val_error = errors[r,f]
                self.ridge_param = self._ridge_params[r]
                removed_input = self._selected_inputs.pop(f)
                if self.verbose:
                    print 'Removed input',removed_input, 'and ridge_param', self.ridge_param, 'with a val. error of', self.val_error, 'Total nr. of selected inputs =', len(self._selected_inputs)-self.with_bias
            else:
                break

        if self.verbose:
            print 'Total time:', time.time()-t_start, 's'

        self._final_training()
        self._clear_memory()

    def _calc_mse_bfs(self, train, val):
        s = self._selected_inputs
        D_t, C_t = la.eigh(self._get('xTx', train)[s,:][:,s] + np.eye(len(s))) #reduce condition number and calculate eigen decomposition
        D_t -= 1
        D_t[np.where(D_t<0)] = 0 #eigenvalues can only be positive, occurs with ill-conditioned matrices
        D_t = 1 / (np.atleast_2d(D_t).T + np.atleast_2d(self._ridge_params))
        xTy_Ct = np.dot(C_t.T, self._get('xTy', train)[s,:])
        xTx_Cv = np.dot(C_t.T, np.dot(self._get('xTx', val)[s,:][:,s], C_t))
        xTy_Cv = (2 * np.dot(C_t.T, self._get('xTy', val)[s,:]))
        if self.with_bias:
            C_t = C_t[:-1,:]
        if len(self._ridge_params) < len(s) - self.with_bias:
            C_tT2 = C_t.T**2
        errors = np.zeros((len(self._ridge_params), len(s) - self.with_bias))
        for o in range(self._output_dim):
            W_t = D_t * xTy_Ct[:,o:o+1]
            CW = np.dot(C_t, W_t)
            if len(self._ridge_params) < len(s) - self.with_bias and not self.low_memory:
                #less straight-forward but faster than the next...
                for r in range(len(self._ridge_params)):
                    D_r = D_t[:,r:r+1] #pre-fetch data
                    W_r = W_t[:,r:r+1] - CW[:,r:r+1].T / np.sum(C_tT2 * D_r, axis=0) * C_t.T * D_r
                    errors[r, :] += np.sum(W_r * (np.dot(xTx_Cv, W_r) - xTy_Cv[:,o:o+1]), axis=0)
            else:
                for f in range(len(s) - self.with_bias):
                    C_ft = C_t[f:f+1,:].T #pre-fetch data
                    uCD = C_ft * D_t
                    W_r = W_t - CW[f:f+1,:] / np.dot(C_ft.T, uCD) * uCD
                    errors[:, f] += np.sum(W_r * (np.dot(xTx_Cv, W_r) - xTy_Cv[:,o:o+1]), axis=0)
        return (errors + np.sum(self._get('yTy', val))) / self._get('len', val) / self._output_dim

    def _calc_other_error_bfs(self, train, val):
        s = self._selected_inputs
        xTx_t = self._get('xTx', train)[s,:][:,s]
        xTy_t = self._get('xTy', train)[s]
        x = self._get('x', val)[:,s]
        y = self._get('y', val)
        errors = np.zeros((len(self._ridge_params), len(s) - self.with_bias))
        for i in range(len(s) - self.with_bias):
            a = range(len(s) - self.with_bias) #tested inputs
            a.pop(i)
            for r in range(len(self._ridge_params)):
                output = np.dot(x[:,a], la.solve(xTx_t[a,:][:,a] + self._ridge_params[r] * np.eye(len(a)), xTy_t[a,:]))
                for o in range(self._output_dim):
                    errors[r,i] += self.other_error_measure(output[:,o], y[:,o])
        errors[np.where(errors<0)] = np.nan
        return errors / self._output_dim

    def _final_training(self):
        # Calculate final weights
        s = self._selected_inputs
        xTx = self._get('xTx', range(len(self._xTx_list)))[s,:][:,s]
        xTy = self._get('xTy', range(len(self._xTx_list)))[s,:]
        W = la.solve(xTx + self.ridge_param * np.eye(len(xTx)), xTy)
        self.w = np.zeros((self._input_dim, self._output_dim))
        if self.with_bias:
            self.b = W[-1, :]
            self.w[s[:-1],:] = W[:-1, :]
            self._selected_inputs = s[:-1]
        else:
            self.b = 0
            self.w[s,:] = W

class BFSClassReweightedRidgeRegressionNode(BFSRidgeRegressionNode, ClassReweightedRidgeRegressionNode):
    pass


class FFSRidgeRegressionNode(BFSRidgeRegressionNode):
    '''
    Forward Feature Selection Ridge Regression Node

    Multiple outputs are not independently optimized...
    '''
    def _stop_training(self):
        if not type(self.ridge_param) is list and not type(self.ridge_param) is np.ndarray:
            self._ridge_params = [self.ridge_param]
        else:
            self._ridge_params = self.ridge_param

        import time
        t_start = time.time()
        if self.other_error_measure:
            calc_error = self._calc_other_error_ffs
        else:
            calc_error = self._calc_mse_ffs
        if self.with_bias:
            self._selected_inputs = [self._input_dim] #select bias
        else:
            self._selected_inputs = [] #selected inputs
        train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
        while len(self._selected_inputs) < (self._input_dim + self.with_bias):
            errors = np.zeros((len(self._ridge_params), self._input_dim + self.with_bias - len(self._selected_inputs)))
            val_sets = range(len(train_samples))
            if self.verbose:
                val_sets = mdp.utils.progressinfo(val_sets, style='timer')
            for k in val_sets:
                errors += calc_error(train_samples[k], val_samples[k])
            errors /= len(train_samples)

            r, f = np.where(errors == np.nanmin(errors))
            r, f = r[-1], f[0]
            if len(self._selected_inputs)==0 or (self.with_bias and len(self._selected_inputs)==1) or errors[r,f] < self.val_error:
                self.val_error = errors[r,f]
                self.ridge_param = self._ridge_params[r]
                us = range(self._input_dim + self.with_bias) #unselected inputs
                for i in self._selected_inputs:
                    us.remove(i)
                self._selected_inputs.append(us[f])
                if self.verbose:
                    print 'Selected input', us[f], 'and ridge_param', self.ridge_param, ' with a val. error of', self.val_error, 'Total nr. of selected inputs =', len(self._selected_inputs)-self.with_bias
            else:
                self._selected_inputs.sort()
                break

        if self.verbose:
            print 'Total time:', time.time()-t_start, 's'

        self._final_training()
        self._clear_memory()

    def _calc_mse_ffs(self, train, val):
        '''Calculate the mse for this validation set'''
        s = self._selected_inputs
        #Perform eigen decomposition
        xTx_t = self._get('xTx', train)
        D_t = np.zeros((self._input_dim + self.with_bias, 1))
        C_t = np.eye(len(D_t))
        if len(s)>0:
            D_t[0:len(s),0], C_t[0:len(s),:len(s)] = la.eigh(xTx_t[s,:][:,s] + np.eye(len(s))) #reduce condition number
            D_t[0:len(s)] -= 1
            D_t[np.where(D_t<0)] = 0
        D_t = 1 / (D_t + np.atleast_2d(self._ridge_params))
        #Reorder and compute covariance matrices
        us = range(len(D_t)) #unselected inputs
        for i in s:
            us.remove(i)
        a = copy.copy(s) #reordered inputs
        a.extend(us)
        xTx_t = xTx_t[a,:][:,a]
        xTy_Ct = np.dot(C_t.T, self._get('xTy', train)[a,:])
        xTx_Cv = np.dot(C_t.T, np.dot(self._get('xTx', val)[a,:][:,a], C_t))
        xTy_Cv = 2 * np.dot(C_t.T, self._get('xTy', val)[a,:])
        #Prepare base output weights
        W_t=[]
        for o in range(self._output_dim):
            W_t.append(D_t * xTy_Ct[:,o:o+1])
        D_t = D_t[:len(s)+1,:]
        #Create a rank 2 update matrix and pre-compute all non trivial elements of uTC matrix
        R = np.array([[0,1], [1,0]])
        C_u = np.dot(xTx_t[:len(s),len(s):].T, C_t[:len(s),:len(s)])
        C_u = np.concatenate((C_u, np.atleast_2d(np.diag(xTx_t)[len(s):]).T / 2), axis=1)
        #Create empty uTC matrix
        C_uf = np.zeros((2, len(s)+1))
        C_uf[0,-1] = 1
        #Test added features and ridge params and average over outputs
        errors = np.zeros((len(self._ridge_params), self._input_dim + self.with_bias - len(s)))
        W_te = np.empty((len(s)+1, len(self._ridge_params)))
        a = range(len(s)+1) #test inputs
        for f in range(self._input_dim + self.with_bias - len(s)):
            a[-1] = len(s) + f
            C_uf[1,:] = C_u[f,:]
            for o in range(self._output_dim):
                W_to = W_t[o][a,:]
                for r in range(len(self._ridge_params)):
                    C_ufD = C_uf * D_t[:,r:r+1].T
                    try:
                        W_te[:,r:r+1] = W_to[:,r:r+1] - np.dot(C_ufD.T, np.dot(la.inv(R + np.dot(C_ufD, C_uf.T)), np.dot(C_uf, W_to[:,r:r+1])))
                    except la.LinAlgError: #avoids singular matrix inversion errors
                        W_te[:,r] = np.nan
                errors[:,f] += np.sum(W_te * (np.dot(xTx_Cv[a,:][:,a], W_te) - xTy_Cv[a,o:o+1]), axis=0)
        return (errors + np.sum(self._get('yTy', val))) / self._get('len', val) / self._output_dim


    def _calc_other_error_ffs(self, train, val):
        # Calculate error for this validation set using other error measure
        errors = np.zeros((len(self._ridge_params), self._input_dim + self.with_bias - len(self._selected_inputs)))
        xTx_t = self._get('xTx', train)
        xTy_t = self._get('xTy', train)
        x = self._get('x', val)
        y = self._get('y', val)

        us = range(self._input_dim + self.with_bias) #unselected inputs
        for i in self._selected_inputs:
            us.remove(i)
        a = copy.copy(self._selected_inputs) #test inputs
        a.append(-1)

        for i in range(len(us)):
            a[-1] = us[i]
            for r in range(len(self._ridge_params)):
                try:
                    output = np.dot(x[:,a], la.solve(xTx_t[a,:][:,a] + self._ridge_params[r] * np.eye(len(a)), xTy_t[a,:]))
                    for o in range(self._output_dim):
                        errors[r,i] += self.other_error_measure(output[:,o], y[:,o])
                except la.LinAlgError: #avoids singular matrix inversion errors
                    errors[r,i] = np.nan
        return errors / self._output_dim


class FFSClassReweightedRidgeRegressionNode(FFSRidgeRegressionNode, ClassReweightedRidgeRegressionNode):
    pass

def lars_path_copy(X, y, Xy=None, Gram=None, max_features=None,
              alpha_min=0, method='lar', overwrite_X=False,
              overwrite_Gram=False, verbose=False):

    """ Copy made to avoid bugs in the scikits.learn toolbox

        Compute Least Angle Regression and LASSO path

        Parameters
        -----------
        X: array, shape: (n_samples, n_features)
            Input data

        y: array, shape: (n_samples)
            Input targets

        max_features: integer, optional
            Maximum number of selected features.

        Gram: array, shape: (n_features, n_features), optional
            Precomputed Gram matrix (X' * X)

        alpha_min: float, optional
            Minimum correlation along the path. It corresponds to the
            regularization parameter alpha parameter in the Lasso.

        method: 'lar' | 'lasso'
            Specifies the returned model. Select 'lar' for Least Angle
            Regression, 'lasso' for the Lasso.

        Returns
        --------
        alphas: array, shape: (max_features + 1,)
            Maximum of covariances (in absolute value) at each
            iteration.

        active: array, shape (max_features,)
            Indices of active variables at the end of the path.

        coefs: array, shape (n_features, max_features+1)
            Coefficients along the path

        See also
        --------
        :ref:`LassoLARS`, :ref:`LARS`

        Notes
        ------
        * http://en.wikipedia.org/wiki/Least-angle_regression

        * http://en.wikipedia.org/wiki/Lasso_(statistics)#LASSO_method
    """

    import numpy as np
    from scipy import linalg
    from scipy.linalg.lapack import get_lapack_funcs

    from scikits.learn.linear_model.base import LinearModel
    from scikits.learn.utils import arrayfuncs

    n_features = X.shape[1]
    n_samples = len(y)

    if max_features is None:
        max_features = min(n_samples, n_features)

    coefs = np.zeros((max_features + 1, n_features))
    alphas = np.zeros(max_features + 1)
    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False
    eps = np.finfo(X.dtype).eps

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    L = np.empty((max_features, max_features), dtype=X.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    potrs, = get_lapack_funcs(('potrs',), (X,))

    if Gram is None:
        if not overwrite_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    else:
        if not overwrite_Gram:
            Gram = Gram.copy()

    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()

    if verbose:
        print "Step\t\tAdded\t\tDropped\t\tActive set size\t\tC"

    while 1:

        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
            C = np.fabs(C_)
            # to match a for computing gamma_
        else:
            if Gram is None:
                C -= gamma_ * np.abs(np.dot(X.T[0], eq_dir))
            else:
                C -= gamma_ * np.abs(np.dot(Gram[0], least_squares))

        alphas[n_iter] = C / n_samples

        # Check for early stopping
        if alphas[n_iter] < alpha_min: # interpolate
            # interpolation factor 0 <= ss < 1
            ss = (alphas[n_iter-1] - alpha_min) / (alphas[n_iter-1] -
                                                   alphas[n_iter])
            coefs[n_iter] = coefs[n_iter-1] + ss*(coefs[n_iter] -
                            coefs[n_iter-1])
            alphas[n_iter] = alpha_min
            break

        if n_active == max_features:
            break

        if not drop:

            # Update the Cholesky factorization of (Xa * Xa') #
            #                                                 #
            #            ( L   0 )                            #
            #     L  ->  (       )  , where L * w = b         #
            #            ( w   z )    z = 1 - ||w||           #
            #                                                 #
            #   where u is the last added to the active set   #

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx+n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov = Cov[1:] # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active])**2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            arrayfuncs.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active])
            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active,  n_active] = diag

            active.append(indices[n_active])
            n_active += 1

            if verbose:
                print "%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                            n_active, C)

        # least squares solution
        least_squares, info = potrs(L[:n_active, :n_active],
                               sign_active[:n_active], lower=True)

        # is this really needed ?
        AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))
        least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)


        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir))
        gamma_ = min(g1, g2, C/AA)

        # TODO: better names for these variables: z
        drop = False
        z = - coefs[n_iter, active] / least_squares
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:

            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso': gamma_ = z_pos
            drop = True

        n_iter += 1

        if n_iter >= coefs.shape[0]:
            # resize the coefs and alphas array
            add_features = 2 * (max_features - n_active)
            coefs.resize((n_iter + add_features, n_features))
            alphas.resize(n_iter + add_features)

        if n_active == max_features:
            break
        coefs[n_iter, active] = coefs[n_iter-1, active] + \
                                gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        if n_active > n_features:
            break

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            arrayfuncs.cholesky_delete(L[:n_active, :n_active], idx)

            n_active -= 1
            m, n = idx, n_active
            drop_idx = active.pop(idx)

            if Gram is None:
                # propagate dropped variable
                for i in range(idx, n_active):
                    X.T[i], X.T[i+1] = swap(X.T[i], X.T[i+1])
                    indices[i], indices[i+1] =  \
                                indices[i+1], indices[i] # yeah this is stupid

                # TODO: this could be updated
                residual = y - np.dot(X[:, :n_active],
                                      coefs[n_iter, active])
                temp = np.dot(X.T[n_active], residual)

                Cov = np.r_[temp, Cov]
            else:
                for i in range(idx, n_active):
                    indices[i], indices[i+1] =  \
                                indices[i+1], indices[i]
                    Gram[i], Gram[i+1] = swap(Gram[i], Gram[i+1])
                    Gram[:, i], Gram[:, i+1] = swap(Gram[:, i], Gram[:, i+1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
                residual = y - np.dot(X, coefs[n_iter])
                temp = np.dot(X.T[drop_idx], residual)
                Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.) # just to maintain size
            if verbose:
                print "%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp))

    # resize coefs in case of early stop
    alphas = alphas[:n_iter+1]
    coefs = coefs[:n_iter+1]

    return alphas, active, coefs.T


class LARSNode(RidgeRegressionNode):
    '''
    Least Angular Regression Node
    '''
    def __init__(self, method='lar', *args, **kwargs):
        '''
        method can either be 'lar' or 'lasso'. lar is faster, but lasso often performs better.

        For other arguments see RidgeRegressionNode.
        '''
        super(LARSNode,self).__init__(*args, **kwargs)
        if not self.other_error_measure and method=='lasso':
            self.other_error_measure = 'mse_lasso'
            self._x_list = []
            self._y_list = []
        if not method == 'lar' and not method == 'lasso':
            import warnings
            warnings.warn('No correct method given, lar chosen instead. Set method to lar or lasso to avoid this warning.')
            method = 'lar'
        self.method = method

    def _stop_training(self):
        if self.other_error_measure and not isinstance(self.other_error_measure, str):
            calc_error = self._calc_other_error_lars
        else:
            calc_error = self._calc_mse_lars
        train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
        errors = np.zeros((self._input_dim + self.with_bias, self._output_dim))
        val_sets = range(len(train_samples))
        if self.verbose:
            val_sets = mdp.utils.progressinfo(val_sets, style='timer')
        for k in val_sets:
            errors += calc_error(train_samples[k], val_samples[k])

        if self.plot_errors:
            import pylab
            pylab.plot(errors)
            pylab.show()

        self.coef_paths = self._lars_path(range(len(self._len_list)))
        self.w = np.zeros((self._input_dim, self._output_dim))
        self.b = np.zeros((self._output_dim))
        j=np.argmin(errors, axis=0)
        for o in range(self._output_dim):
            self.w[:,o] = self.coef_paths[o][:-1, j[o]]
            self.b[o] = self.coef_paths[o][-1, j[o]]

        self._clear_memory()

    def _calc_mse_lars(self, train, val):
        errors = np.zeros((self._input_dim + self.with_bias, self._output_dim))
        coef_paths = self._lars_path(train)
        xTx_v = self._get('xTx', val)
        xTy_v = 2 * self._get('xTy', val)
        for o in range(self._output_dim):
            errors[:,o] = np.sum(coef_paths[o] * (np.dot(xTx_v, coef_paths[o])  - xTy_v[:,o:o+1]), axis=0)
        return (errors + self._get('yTy', val)) / self._get('len', val)

    def _calc_other_error_lars(self, train, val):
        errors = np.zeros((self._input_dim + self.with_bias, self._output_dim))
        coef_paths = self._lars_path(train)
        x = self._get('x', val)
        y = self._get('y', val)
        for o in range(self._output_dim):
            output = np.dot(x,coef_paths[o])
            for i in range(self._input_dim + self.with_bias):
                errors[i,o] = self.other_error_measure(output[:,i], y[o])
        return errors

    def _lars_path(self, train):
        if self.method == 'lasso':
            X = self._get('x', train)
            y = self._get('y', train)
        else:
            X=np.empty((0, self._input_dim + self.with_bias), dtype=self.dtype)
            y=np.empty((self._get('len',train), 0), dtype=self.dtype)
        Xy = self._get('xTy', train)
        Gram = self._get('xTx', train)
        coef_paths = []
        for o in range(self._output_dim):
            if self.method == 'lasso':
                _, _, cp = lars_path_copy(X=X, y=y[:,o], Xy=Xy[:,o], Gram=Gram, method=self.method, overwrite_X=self._output_dim==1, overwrite_Gram=self._output_dim==1, verbose=False)
            else:
                _, _, cp = lars_path_copy(X=X, y=y, Xy=Xy[:,o], Gram=Gram, method=self.method, overwrite_X=self._output_dim==1, overwrite_Gram=self._output_dim==1, verbose=False)
            #Find the weights that add a feature
            weights = np.zeros((self._input_dim + self.with_bias, self._input_dim + self.with_bias))
            j, k = 0, -1
            while j < cp.shape[1]:
                if k < np.sum(np.abs(cp[:,j:j+1])>0):
                    k+=1
                    weights[:,k] = cp[:,j]
                j+=1
            coef_paths.append(weights)
        return coef_paths


class ClassReweightedLARSNode(LARSNode, ClassReweightedRidgeRegressionNode):
    pass

class OPRidgeRegressionNode(BFSRidgeRegressionNode, LARSNode):
    '''
    Optimally Pruned Ridge Regression Node
    '''
    def __init__(self, ridge_param=mdp.numx.power(10, mdp.numx.arange(-15,15,0.2)), verbose=False, *args, **kwargs):
        '''
        see RidgeRegressionNode
        '''
        super(OPRidgeRegressionNode, self).__init__(ridge_param=ridge_param, verbose=verbose, *args, **kwargs)

    def _stop_training(self):
        if not type(self.ridge_param) is list and not type(self.ridge_param) is np.ndarray:
            self._ridge_params = [self.ridge_param]
        else:
            self._ridge_params = self.ridge_param

        if self._output_dim > 1:
            raise Exception('Only one output is supported')
        if self.other_error_measure:
            calc_error = self._calc_other_error
        else:
            calc_error = self._calc_mse

        ranking = self._get_ranking()

        train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
        errors = np.zeros((len(self._ridge_params), len(ranking)))
        val_sets = range(len(train_samples))
        if self.verbose:
            val_sets = mdp.utils.progressinfo(val_sets, style='timer')
        for k in val_sets:
            for i in range(len(ranking)):
                errors[:,i] += np.sum(calc_error(train_samples[k], val_samples[k], ranking[i]), axis=1)
        errors /= len(train_samples)

        r,i = np.where(errors == np.nanmin(errors))
        r, i = r[-1], i[0]
        self.val_error = errors[r,i]
        self.ridge_param = self._ridge_params[r]
        self._selected_inputs = ranking[i]
        if self.verbose:
            print 'Found a ridge_par =', self.ridge_param, ' with a validation error of:', self.val_error, 'and selected', i+1, 'features.'
        if self.plot_errors:
            import pylab
            pylab.plot(range(1, errors.shape[1]+1), np.nanmin(errors, axis=0))
            pylab.figure()
            pylab.plot(np.log10(self._ridge_params),np.nanmin(errors, axis=1))
            pylab.show()

        self._final_training()
        self._clear_memory()

    def _get_ranking(self):
        path = self._lars_path(range(len(self._xTx_list)))[0]
        ranking = []
        for i in range(1, len(path)):
            ranking.append(np.nonzero(path[:,i])[0])
        return ranking

class ClassReweightedOPRidgeRegressionNode(OPRidgeRegressionNode, ClassReweightedRidgeRegressionNode):
    pass


class BayesianWeightedRegressionNode(RidgeRegressionNode):

    def __init__(self, mu=None, threshold=0.00001, max_iter=100, with_bias=True, clear_memory=True, input_dim=None, output_dim=None, dtype=None):
        super(BayesianWeightedRegressionNode, self).__init__(with_bias=with_bias, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.mu = mu
        self.threshold=threshold
        self.max_iter = max_iter
        self.clear_memory = clear_memory

    def _get_one(self, name, i):
        return self.beta[i] * super(BayesianWeightedRegressionNode, self)._get_one(name, i)

    def _stop_training(self):
        self.alpha = 0
        self.beta = np.ones((len(self._xTx_list)))

        w = np.zeros((self.output_dim,1))
        w_prev = np.zeros((self.output_dim,1))
        if self.mu is None:
            self.mu = np.zeros((self.output_dim,1))

        n_iter = 0
        while n_iter < self.max_iter or np.mean(np.abs(w - w_prev) / np.abs(w_prev)) > self.threshold:
            w_prev = w
            S_n = la.inv(self.alpha * np.eye(self._input_dim + self.with_bias) + self._get('xTx', range(len(self._xTx_list))))
            w = np.dot(S_n, self._get('xTy', range(len(self._xTx_list))))
            self.alpha = (self.input_dim + self.with_bias) / (np.trace(S_n) + np.dot((w - self.mu).T, w - self.mu))
            for i in range(len(self._xTx_list)):
                e = self._get_one('yTy',i) + np.dot(w.T, np.dot(self._get_one('xTx',i), w) - 2 * self._get_one('xTy',i))
                self.beta[i] = self._len_list[i] / (np.trace(np.dot(S_n, self._get_one('xTx', i) / self.beta[i])) + e / self.beta[i])
            n_iter += 1

        if self.with_bias:
            self.w = w[:-1]
            self.b = w[-1]
        else:
            self.w = w
            self.b = 0
        self._clear_memory()

class ParallelLinearRegressionNode(mdp.parallel.ParallelExtensionNode, RidgeRegressionNode):
    """Parallel extension for the LinearRegressionNode and all its derived classes
    (eg. RidgeRegressionNode).
    """
    def _fork(self):
        return self._default_fork()

    def _join(self, forked_node):
        if self._xTx is None:
            self._xTx = forked_node._xTx
            self._xTy = forked_node._xTy
            self._tlen = forked_node._tlen
        else:
            self._xTx += forked_node._xTx
            self._xTy += forked_node._xTy
            self._tlen += forked_node._tlen
