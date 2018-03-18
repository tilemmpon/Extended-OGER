import Oger
import mdp

def test_n_fold_random_shape():
    '''Test if n_fold_random result has desired shape
    '''
    [tr, te] = Oger.evaluation.n_fold_random(4,3)
    assert all(map(lambda(x): tr[1].shape==x.shape, tr))
    assert all(map(lambda(x): te[1].shape==x.shape, te))
    assert len(tr) == 3 
    assert len(te) == 3 

def test_n_fold_random_different():
    '''Test if n_fold_random is all different 
    '''
    tr = Oger.evaluation.n_fold_random(4,3)[0]
    for i,t1 in enumerate(tr):
        for t2 in tr[i+1:]:
            assert set(t1) != set(t2)

def test_train_test_only():
    '''Test if train_test_only gives expected results 
    '''
    [tr, te] = Oger.evaluation.train_test_only(5,.5)
    assert len(tr) == 1
    assert len(tr) == 1
    assert tr[0].shape[0] == 3
    assert te[0].shape[0] == 2

def test_leave_one_out():
    '''Test if leave_one_out gives expected results 
    '''
    [tr, te] = Oger.evaluation.leave_one_out(5)
    assert len(tr) == len(te) == 5

def test_optimizer_no_last_data_iterable():
    '''Test if the optimizer complains if no data is given for the last node in the flow even if
    it is not trainable (needed for validation). 
    '''
    inputs, outputs = Oger.datasets.narma30()

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode()
    readout = Oger.nodes.RidgeRegressionNode()
    tnode = Oger.nodes.ThresholdNode()
    
    data = [[], zip(inputs, outputs), None]

    flow = mdp.Flow([reservoir, readout, tnode])
    try:
        Oger.evaluation.validate(data, flow, Oger.utils.nrmse, n_folds =5, progress=False)
    except Exception:
        pass
