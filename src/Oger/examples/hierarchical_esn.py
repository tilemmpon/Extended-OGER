import Oger
import pylab
import mdp

# TODO: these nodes should be part of Oger!
class MultNode(mdp.Node):
    def __init__(self, inputs, votes, dtype='float64'):
        super(MultNode, self).__init__(inputs * votes + votes, inputs, dtype)
        
        self.inputs = inputs
        self.votes = votes
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        hyp = mdp.numx.reshape(x[:, 0:self.inputs * self.votes], (self.inputs, self.votes))
        vote = x[:, self.inputs * self.votes:]

        return mdp.numx.dot(hyp, vote.T).T

class GradientMultNode(Oger.gradient.GradientExtensionNode, MultNode):
    def _calculate_gradient(self, y):
        hyp = mdp.numx.reshape(self._last_x[:, 0:self.inputs * self.votes], (self.inputs, self.votes))
        vote = self._last_x[:, self.inputs * self.votes:]

        dx0 = mdp.numx.reshape(mdp.numx.outer(y, vote), (1, self.inputs * self.votes))
        dx1 = mdp.numx.dot(y, hyp)
        
        return mdp.numx.concatenate((dx0, dx1), axis=1)

# TODO: this is really ugly, we should find a clean solution for this
class ConcatFeedbackNode(Oger.nodes.FeedbackNode):
    def __iter__(self):
        if isinstance(self.input, mdp.numx.ndarray): 
            while self.current_timestep < self.n_timesteps:            
                yield mdp.numx.concatenate((self.input[self.current_timestep:self.current_timestep + 1, :], self.last_value), axis=1)
                self.current_timestep += 1
        else:
            while self.current_timestep < self.n_timesteps:
                prev_data = self.input.__iter__().next()
                yield mdp.numx.concatenate((prev_data, self.last_value), axis=1)
                self.current_timestep += 1

class GradientFeedbackNode(Oger.gradient.GradientExtensionNode, Oger.nodes.FeedbackNode):
    def _calculate_gradient(self, y):
        # TODO        
        return y

class GradientReservoirNode(Oger.gradient.GradientExtensionNode, Oger.nodes.ReservoirNode):
    def _calculate_gradient(self, y):
        # TODO: not exactly correct!
        
        dx = mdp.numx.dot(y, self.w_in)
        return dx

if __name__ == "__main__":

    x = Oger.datasets.mso(sample_len=10)
    x = mdp.numx.concatenate((x[0], x[0]), axis=1)
    y = x[1:, :]
    x = x[0:-1, :]
    
    # construct individual nodes
    reservoir1 = Oger.nodes.FeedbackReservoirNode(input_dim=x.shape[1], output_dim=100, input_scaling=0.1)
    reservoir1.reset_states = False
    readout1 = Oger.nodes.PerceptronNode(100, 20)

    reservoir2 = Oger.nodes.FeedbackReservoirNode(input_dim=20, output_dim=100, input_scaling=0.1)
    reservoir2.reset_states = False
    readout2 = Oger.nodes.PerceptronNode(100, 10, transfer_func=Oger.utils.LogisticFunction)
    
    #fbnode = ConcatFeedbackNode(input_dim=20, n_timesteps=999)
    #fbnode.input = x
    fbnode = Oger.nodes.FeedbackNode(input_dim=20, n_timesteps=x.shape[0])
    fbnode.last_value = mdp.numx.zeros((1, 20)) 

    layer = mdp.hinet.Layer([mdp.hinet.FlowNode(reservoir1 + readout1 + fbnode), mdp.hinet.FlowNode(reservoir2 + readout2)])

    flow = layer + MultNode(2, 10)

    bpnode = Oger.gradient.BackpropNode(flow, Oger.gradient.GradientDescentTrainer() , loss_func=Oger.utils.mse, n_epochs=5)
    bpflow = mdp.Flow([bpnode, ])
    
    reservoir1.states = mdp.numx.zeros((1, 100))
    reservoir2.states = mdp.numx.zeros((1, 100))
    
    #fbnode.reset()
    bpflow.train([Oger.utils.ConcatenatingIterator([fbnode, x], y)])
        
#        for i, data in enumerate(fbnode):
#            bpnode.train(x = data + mdp.numx.random.randn(data.shape[0], data.shape[1])*0.01, t = y[i, :])
#            bpnode.train(x = data, t = y[i, :])

    fbnode.reset()        
    out = flow.execute(fbnode)

    fbnode2 = Oger.nodes.FeedbackNode(input_dim=2, n_timesteps=1000)
    fbnode2.last_value = out[-1:, :]

    flow += fbnode2

    fbnode.reset()        
    fbnode.n_timesteps = 1000    
    fbnode.input = fbnode2
    
    result = flow.execute(fbnode)
    
    pylab.plot(result)
    pylab.show()
    


