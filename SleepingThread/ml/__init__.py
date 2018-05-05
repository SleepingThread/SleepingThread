# -*- coding: utf-8 -*-

"""

"""

import numpy as np

def weighted_accuracy(estimator,X,y):
    """
    y - two class prediction
    """
    y = np.array(y)
    prediction = estimator.predict(X)
    
    # work with labels
    labels = []
    for el in y:
        if el not in labels:
            labels.append(el)
            
    labels_qual = {el:0.0 for el in labels}
    labels_size = {el:np.sum(y==el) for el in labels}
    for i in xrange(len(y)):  
        labels_qual[y[i]] += (y[i]==prediction[i])
    
    qual = 0.0
    for key in labels_qual:
        qual += labels_qual[key]/labels_size[key]
    
    qual = qual/len(labels)
    
    return qual

def getNextVec(itvec,itend):
    """
    itvec,itend - array
    """
    for i in xrange(len(itvec)):
        itvec[i] += 1
        if itvec[i]<itend[i]:
            break
        else:
            itvec[i] = 0
   
    if not (i==len(itvec)-1 and itvec[-1]==0):
        return True
    else:
        return False

def getNextDict(itvec,itend,keys):
    """
    itvec,itend - dictionaries
    """
    i = 0
    for key in keys:
        i += 1
        itvec[key] += 1
        if itvec[key]<itend[key]:
            break
        else:
            itvec[key] = 0
   
    if not (i==len(keys) and itvec[keys[-1]]==0):
        return True
    else:
        return False

def getNext(itvec,itend,keys=None):
    if keys is None:
        return getNextVec(itvec,itend)
    else:
        return getNextDict(itvec,itend,keys)

    return

def ExecuteGrid(func,params):
    """
    params - dictionary
    """
    itvec = {key:0 for key in params}
    itend = {key:len(params[key]) for key in params}
    keys = params.keys()
    not_end = True
    while not_end:
        kwargs = {key:params[key][itvec[key]] for key in params}
        not_end = getNext(itvec,itend,keys)

        func(**kwargs)

    return

