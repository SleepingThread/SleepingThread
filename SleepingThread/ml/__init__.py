# -*- coding: utf-8 -*-

"""

"""

import sys
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

#========================================================
# Search best parameter with scree
#========================================================

from sklearn.base import clone

class ScreeSearch(object):
    def __init__(self,estimator,params,mode="quality"):
        """
        mode - "error" | "quality"
        params - {"param_name":list of values}
        """

        self.mode = mode

        # params list for estimator
        self.params = params
        self.estimator = estimator

        # store best trainer
        self.trainer_params = None
        self.trainer = None

        self.scores = []

        if len(params[params.keys()[0]]) <=3:
            raise Exception("Not enough params")

        return

    def fit(self,X,y=None,scoring=None):
        """
        scoring - dict with metrics(estimator,X,y)
        """

        self.scores = []
        param_name = self.params.keys()[0]
        for el in self.params[param_name]:
            cur_trainer = clone(self.estimator)
            args = {param_name:el}
            cur_trainer.set_params(**args)
            
            cur_trainer.fit(X,y)

            if scoring is None:
                self.scores.append(cur_trainer.score(X,y))
            else:
                scores = {}
                key = scoring.keys()[0]
                self.scores.append(scoring[key](cur_trainer,X,y))

        # find best parameter
        scree_vals = []
        for i in xrange(1,len(self.params[param_name])-1):
            a = self.scores[i-1]
            b = self.scores[i]
            c = self.scores[i+1]
            scree_vals.append(a+c-2*b)

        if self.mode == "error":
            # find max if self.scores - error
            best_ind = np.argmax(scree_vals)+1

        elif self.mode == "quality":
            # find min if self.scores - quality
            best_ind = np.argmin(scree_vals)+1

        self.trainer_params = {param_name:self.params[param_name][best_ind]}
        self.trainer = clone(self.estimator)
        self.trainer.set_params(**self.trainer_params)
        self.trainer.fit(X,y)

        return

    def predict(self,X):
        return self.trainer.predict(X)

    def drawScree(self):
        from matplotlib import pyplot as plt
        param_name = self.params.keys()[0]
        plt.plot(self.params[param_name],self.scores)
        plt.show()
        return

class GrowSpeed(object):
    def __init__(self,estimator,params,mode="quality",coef=0.5,verbose=0):
        """
        mode - "error" | "quality"
        params - {"param_name":list of values}
        """

        self.verbose = verbose

        self.coef = coef
        self.mode = mode

        # params list for estimator
        self.params = params
        self.estimator = estimator

        # store best trainer
        self.trainer_params = None
        self.trainer = None

        self.scores = []

        if len(params[params.keys()[0]]) <=3:
            raise Exception("Not enough params")

        return

    def fit(self,X,y=None,scoring=None):
        """
        scoring - dict with metrics(estimator,X,y)
        """

        self.scores = []
        param_name = self.params.keys()[0]
        params = self.params[param_name]

        _counter = 0
        _p_len = len(params)
        for el in params:
            _counter += 1
            if self.verbose > 0:
                sys.stdout.write("\rGrowSpeed: process "+str(_counter)+" from "+str(_p_len))

            cur_trainer = clone(self.estimator)
            args = {param_name:el}
            cur_trainer.set_params(**args)
            
            cur_trainer.fit(X,y)

            if scoring is None:
                self.scores.append(cur_trainer.score(X,y))
            else:
                scores = {}
                key = scoring.keys()[0]
                self.scores.append(scoring[key](cur_trainer,X,y))

        # find best parameter
        grow_vals = []
        for i in xrange(1,len(params)-1):
            a = self.scores[i-1]
            c = self.scores[i+1]
            grow_vals.append((c-a)/(params[i+1]-params[i-1]))

        if self.verbose > 0:
            sys.stdout.write("\n")

        if self.mode == "error" or self.mode == "quality":
            # find first which abs(.) less than 
            # abs((self.scores[-1]+self.scores[0])/(params[-1]-params[0])) 
            val = self.coef*abs((self.scores[-1]+self.scores[0])/(params[-1]-params[0]))
            for ind,el in enumerate(grow_vals):
                if abs(el)<val:
                    best_ind = ind + 1
                    break

        self.trainer_params = {param_name:self.params[param_name][best_ind]}
        self.trainer = clone(self.estimator)
        self.trainer.set_params(**self.trainer_params)
        self.trainer.fit(X,y)

        return

    def predict(self,X):
        return self.trainer.predict(X)

    def drawScores(self):
        from matplotlib import pyplot as plt
        param_name = self.params.keys()[0]
        plt.plot(self.params[param_name],self.scores)
        plt.show()
        return


#========================================================
# SVM with zero norm from Alexey Shestov <github: nelenivy>
#    Realization of Jason Weston "Use of the Zero-Norm with Linear Models and Kernel Methods"
#========================================================
