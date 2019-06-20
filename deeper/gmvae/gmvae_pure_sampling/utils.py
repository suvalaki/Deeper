import numpy as np
from sklearn import metrics

def chain_call(func,x, num):
    iters = x.shape[0]//num
    
    result = []
    j=0
    while j*num < x.shape[0]:
        result.append(func(x[j*num:min((j+1)*num, x.shape[0])]))
        j+=1
    
    num_dats = len(result[0])
    #pivot the resultsif
    if type(result[0]) == tuple:
        result_pivot = [ np.concatenate([y[i] for y in result], -1) for i in range(num_dats) ]
    else:
        result_pivot=np.concatenate(result, -1)
    return result_pivot



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
