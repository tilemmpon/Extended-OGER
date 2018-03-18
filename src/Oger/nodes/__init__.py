"""
This subpackage contains a number of Oger-related nodes. It contains several additional MDP nodes such as RBM nodes, reservoir nodes and signal processing nodes.
"""

from flows import (FreerunFlow)
from reservoir_nodes import (ReservoirNode, LeakyReservoirNode, SparseReservoirNode, SparseLeakyReservoirNode, TrainableReservoirNode, HebbReservoirNode, GaussianIPReservoirNode, BandpassReservoirNode, CUDAReservoirNode,
ExtendedReservoirNode, CyclicSORMsReservoir, SparseAndOrthogonalMatricesReservoir, FeedForwardESNReservoir,
CycleReservoirWithJumpsNode, SimpleCycleReservoirNode, DelayLineWithFeedbackReservoirNode, DelayLineReservoirNode)
from linear_nodes import (RidgeRegressionNode, ClassReweightedRidgeRegressionNode, BFSRidgeRegressionNode, BFSClassReweightedRidgeRegressionNode, FFSRidgeRegressionNode, FFSClassReweightedRidgeRegressionNode, LARSNode, ClassReweightedLARSNode, OPRidgeRegressionNode, ClassReweightedOPRidgeRegressionNode, BayesianWeightedRegressionNode)
from nonlinear_nodes import (ThresholdNode, PerceptronNode, IRLSLogisticRegressionNode, LogisticRegressionNode)
from rbm_nodes import (ERBMNode, CRBMNode, CUDACRBMNode, CUDATRMNode)
from utility_nodes import (FeedbackNode, MeanAcrossTimeNode, WTANode, ShiftNode, FeedbackShiftNode, ResampleNode, TimeFramesNode2, SupervisedLayer, MaxVotingNode)
from csp_nodes import CSPNode
from elm_nodes import ELMNode
try:
    from spiking_nodes import (BrianIFReservoirNode, GenericSpikingReservoirNode, SpikingRandomIFReservoirNode, SpikingRandomIFDynSynReservoirNode, SpatialSpikingReservoirNode)
    del spiking_nodes
except:
    pass
from layers import (SplitOutputLayer, SplitOutputSameInputLayer)
#from ode_nodes import (OdeNode)

# clean up namespace
del flows
del reservoir_nodes
del linear_nodes
del nonlinear_nodes
del rbm_nodes
del utility_nodes
del layers
del csp_nodes
del elm_nodes


__all__ = [ 'FreerunFlow', 'OnlineFreerunFlow',
            'ReservoirNode', 'LeakyReservoirNode', 'TrainableReservoirNode', 'HebbReservoirNode',  'GaussianIPReservoirNode' ,'BandpassReservoirNode','CUDAReservoirNode',
	'ExtendedReservoirNode', 'CyclicSORMsReservoir', 'SparseAndOrthogonalMatricesReservoir', 'FeedForwardESNReservoir',
'CycleReservoirWithJumpsNode', 'SimpleCycleReservoirNode', 'DelayLineWithFeedbackReservoirNode', 'DelayLineReservoirNode'
            'RidgeRegressionNode', 'ClassReweightedRidgeRegressionNode', 'BFSRidgeRegressionNode', 'BFSClassReweightedRidgeRegressionNode', 'FFSRidgeRegressionNode', 'FFSClassReweightedRidgeRegressionNode', 'LARSNode', 'ClassReweightedLARSNode', 'OPRidgeRegressionNode', 'ClassReweightedOPRidgeRegressionNode', 'BayesianWeightedRegressionNode',
            'ThresholdNode', 'PerceptronNode', 'IRLSLogisticRegressionNode', 'LogisticRegressionNode',
            'ERBMNode', 'CRBMNode', 'CUDACRBMNode', 'CUDAReservoirNode', 'CUDATRMNode',
            'FeedbackNode', 'MeanAcrossTimeNode', 'WTANode', 'ShiftNode', 'FeedbackShiftNode', 'ResampleNode', 'SupervisedLayer', 'MaxVotingNode',
            'GenericSpikingReservoirNode', 'SpikingRandomIFReservoirNode', 'SpikingRandomIFDynSynReservoirNode', 'SpatialSpikingReservoirNode',
            'TimeFramesNode2', 'SplitOutputLayer', 'SplitOutputSameInputLayer'
            'CSPNode',
            'ELMNode']


