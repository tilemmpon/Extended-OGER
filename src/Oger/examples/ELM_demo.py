# This is a demo for an ELM, OP-ELM and RBFS-ELM that tries to model a sinc function
# OP-ELM often performs better but not always
# Regularized Backward Feature Selection ELM is slower but should perform even better...

import mdp
import Oger
import numpy as np
import pylab

X_orr = np.atleast_2d(np.arange(-5, 5, 0.025)).T
Y_orr = np.sinc(X_orr)
#shuffle data and add noise
P = np.random.permutation(len(X_orr))
X, Y = X_orr[P,:], Y_orr[P,:] + (np.random.rand(len(X_orr),1)-.5)/4

#split the data for training
n_folds = 10
l = np.floor(len(X)/n_folds)
x,y=[],[]
for i in range(n_folds-1):
    x.append(X[i*l:(i+1)*l,:])
    y.append(Y[i*l:(i+1)*l,:])

#last fold as test data
x_t = X[(n_folds-1)*l:,:]
y_t = Y_orr[P[(n_folds-1)*l:],:]

#elm node simulates elm without training
elm = Oger.nodes.ELMNode()

#ELM training using Linear Regression (no regularization)
readout = Oger.nodes.RidgeRegressionNode(ridge_param=0)
flow = mdp.Flow([elm, readout],verbose=True)
print '\nTraining ELM...\n'
flow.train([x, zip(x,y)])
yh_t_elm = flow(x_t) #ELM test output

#OP-ELM training using Optimally Pruned Linear Regression (no regularization)
readout = Oger.nodes.OPRidgeRegressionNode(ridge_param=0)
flow = mdp.Flow([elm, readout],verbose=True)
print '\nTraining OP-ELM...\n'
flow.train([x, zip(x,y)])
yh_t_opelm = flow(x_t) #OP-ELM test output

#Backward Feature Selection training using BFSRidgeRegressionNode (with regularization)
readout = Oger.nodes.BFSRidgeRegressionNode(verbose=False)
flow = mdp.Flow([elm, readout],verbose=True)
print '\nTraining Backward Feature Selection with Regularization...\n'
flow.train([x, zip(x,y)])
yh_t_bfs = flow(x_t) #OP-ELM test output


print '\nELM: Test NRMSE =', Oger.utils.nrmse(yh_t_elm, y_t)
print 'OP-ELM: Test NRMSE =', Oger.utils.nrmse(yh_t_opelm, y_t)
print 'RBFS-ELM (BFS with Regularization): Test NRMSE =', Oger.utils.nrmse(yh_t_bfs, y_t), '\n'


pylab.scatter(X,Y, c='k', marker='+')
pylab.plot(X_orr, Y_orr, linewidth=2)
pylab.scatter(x_t,yh_t_elm,c='r',linewidths=2)
pylab.title('ELM')

pylab.figure()
pylab.scatter(X,Y, c='k', marker='+')
pylab.plot(X_orr, Y_orr, linewidth=2)
pylab.scatter(x_t,yh_t_opelm,c='r',linewidths=2)
pylab.title('OP-ELM')

pylab.figure()
pylab.scatter(X,Y, c='k', marker='+')
pylab.plot(X_orr, Y_orr, linewidth=2)
pylab.scatter(x_t,yh_t_opelm,c='r',linewidths=2)
pylab.title('RBFS-ELM (Backward Feature Selection with Regularization)')
pylab.show()